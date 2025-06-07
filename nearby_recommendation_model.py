from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class NearbyWorkspaceRecommender:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection = None
        self.model = None
        self.workspace_data = None
        self.user_data = None
        self.rating_data = None
        
        # Test connection immediately
        self.test_connection()
        
    def test_connection(self):
        """Test database connection on initialization"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                print("✅ Successfully connected to MySQL database")
                self.connection.close()
            else:
                print("❌ Failed to connect to MySQL database")
        except Error as e:
            print(f"❌ MySQL connection error: {e}")
        except Exception as e:
            print(f"❌ Unexpected connection error: {e}")
    
    def connect(self) -> bool:
        """Establish a new database connection"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            return True
        except Error as e:
            print(f"❌ MySQL connection error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected connection error: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        try:
            if self.connection and self.connection.is_connected():
                self.connection.close()
        except Exception as e:
            print(f"❌ Error disconnecting: {e}")
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two geographic points"""
        try:
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            return 6371 * 2 * atan2(sqrt(a), sqrt(1-a))  # km
        except Exception as e:
            print(f"❌ Error calculating distance: {e}")
            return float('inf')  # Return large distance if calculation fails
    
    def load_data(self) -> bool:
        """Load data from database using mysql-connector"""
        if not self.connect():
            return False
            
        try:
            print("⏳ Loading workspace data...")
            workspace_query = """
            SELECT w.Id, w.Name, w.Type, w.AvgRating, 
                   l.Latitude, l.Longitude, l.City
            FROM Workspace w
            JOIN Location l ON w.LocationId = l.Id
            """
            self.workspace_data = pd.read_sql(workspace_query, self.connection)
            print(f"Loaded {len(self.workspace_data)} workspaces")
            
            print("⏳ Loading user data...")
            user_query = "SELECT Id, FName, LName FROM User"
            self.user_data = pd.read_sql(user_query, self.connection)
            print(f"Loaded {len(self.user_data)} users")
            
            print("⏳ Loading rating data...")
            rating_query = """
            SELECT UserId as user_id, WorkspaceId as workspace_id, Rating as rating 
            FROM Review
            WHERE Rating IS NOT NULL
            """
            self.rating_data = pd.read_sql(rating_query, self.connection)
            print(f"Loaded {len(self.rating_data)} ratings")
            
            return True
            
        except Error as e:
            print(f"❌ MySQL error loading data: {e}")
            return False
        except pd.errors.DatabaseError as e:
            print(f"❌ Pandas SQL error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error loading data: {e}")
            return False
        finally:
            self.disconnect()
    
    def recommend_workspaces(self, user_id: str, user_lat: float, user_lon: float, top_n: int = 5) -> List[Dict]:
        """
        Generate personalized workspace recommendations with location filtering
        Returns a list of recommendation dictionaries
        """
        print("\n✨ Starting recommendation process...")
        
        try:
            # 1. Load the data
            if not self.load_data():
                print("❌ Failed to load data")
                return []
            
            # 2. Validate user exists
            user_id = str(user_id).strip()  # Ensure clean user ID
            if user_id not in self.user_data['Id'].astype(str).values:
                print(f"⚠️ User {user_id} not found in database")
                return self.distance_based_recommendations(user_lat, user_lon, top_n)
            
            # 3. If no ratings exist, fall back to distance-based only
            if len(self.rating_data) == 0:
                print("ℹ️ No ratings found - using distance-based recommendations only")
                return self.distance_based_recommendations(user_lat, user_lon, top_n)
            
            # 4. Prepare TensorFlow datasets with proper cleaning
            print("⏳ Preparing data for modeling...")
            
            def clean_ids(id_series):
                """Ensure IDs are properly formatted strings"""
                return (
                    id_series
                    .astype(str)
                    .str.strip()
                    .str.replace(r'[^a-zA-Z0-9_-]', '', regex=True)
                    .values
                )
            
            workspace_ids = clean_ids(self.workspace_data['Id'])
            user_ids = clean_ids(self.user_data['Id'])
            
            # Debug: Print sample IDs
            print(f"Sample workspace IDs: {workspace_ids[:5]}")
            print(f"Sample user IDs: {user_ids[:5]}")
            
            # Validate we have IDs to work with
            if len(workspace_ids) == 0 or len(user_ids) == 0:
                print("⚠️ No valid workspace or user IDs found")
                return self.distance_based_recommendations(user_lat, user_lon, top_n)
            
            # 5. Create embeddings with validation
            print("⏳ Building recommendation model...")
            try:
                workspace_embedding = tf.keras.Sequential([
                    tf.keras.layers.StringLookup(
                        vocabulary=workspace_ids, 
                        mask_token=None,
                        num_oov_indices=0
                    ),
                    tf.keras.layers.Embedding(len(workspace_ids) + 1, 32)
                ])
                
                user_embedding = tf.keras.Sequential([
                    tf.keras.layers.StringLookup(
                        vocabulary=user_ids,
                        mask_token=None,
                        num_oov_indices=0
                    ),
                    tf.keras.layers.Embedding(len(user_ids) + 1, 32)
                ])
            except Exception as e:
                print(f"⚠️ Error creating embeddings: {e}")
                return self.distance_based_recommendations(user_lat, user_lon, top_n)
            
            # 6. Create retrieval model
            retrieval_task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    tf.data.Dataset.from_tensor_slices(workspace_ids)
                    .batch(128)
                    .map(workspace_embedding)
                )
            )
            
            # 7. Create index with error handling
            print("⏳ Creating recommendation index...")
            try:
                index = tfrs.layers.factorized_top_k.BruteForce(user_embedding)
                index.index_from_dataset(
                    tf.data.Dataset.from_tensor_slices(workspace_ids)
                    .batch(100)
                    .map(lambda x: (x, workspace_embedding(x)))
                )
            except Exception as e:
                print(f"⚠️ Error creating index: {e}")
                return self.distance_based_recommendations(user_lat, user_lon, top_n)
            
            # 8. Get recommendations
            print(f"🔍 Finding recommendations for user {user_id}...")
            try:
                _, recommended_ids = index(np.array([user_id]))
                recommended_ids = [
                    id.decode('utf-8') if isinstance(id, bytes) else str(id) 
                    for id in recommended_ids[0].numpy()
                ]
                print(f"Raw recommendations: {recommended_ids[:10]}")
            except Exception as e:
                print(f"⚠️ Error getting recommendations: {e}")
                return self.distance_based_recommendations(user_lat, user_lon, top_n)
            
            # 9. Filter by distance
            print("📍 Filtering by distance...")
            recommendations = []
            for ws_id in recommended_ids:
                try:
                    if ws_id in self.workspace_data['Id'].astype(str).values:
                        ws = self.workspace_data[self.workspace_data['Id'].astype(str) == ws_id].iloc[0]
                        distance = self.haversine_distance(
                            user_lat, user_lon,
                            float(ws['Latitude']), float(ws['Longitude'])
                        )
                        
                        if distance <= 50:  # 50km radius
                            recommendations.append({
                                'workspace_id': ws_id,
                                'name': ws['Name'],
                                'type': ws['Type'],
                                'rating': float(ws['AvgRating']) if not pd.isna(ws['AvgRating']) else None,
                                'distance_km': distance,
                                'city': ws['City']
                            })
                except Exception as e:
                    print(f"⚠️ Error processing workspace {ws_id}: {e}")
                    continue
            
            # 10. Sort by distance and rating
            recommendations = sorted(
                recommendations,
                key=lambda x: (x['distance_km'], -x['rating'] if x['rating'] is not None else 0)
            )[:top_n]
            
            print(f"✅ Found {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            print(f"❌ Critical error during recommendation: {e}")
            return self.distance_based_recommendations(user_lat, user_lon, top_n)
    
    def distance_based_recommendations(self, user_lat: float, user_lon: float, top_n: int = 5) -> List[Dict]:
        """Fallback method when no ratings exist or ML fails"""
        try:
            if self.workspace_data is None:
                return []
                
            print("ℹ️ Using distance-based fallback method")
            recommendations = []
            
            for _, ws in self.workspace_data.iterrows():
                try:
                    distance = self.haversine_distance(
                        user_lat, user_lon,
                        float(ws['Latitude']), float(ws['Longitude'])
                    )
                    
                    if distance <= 50:  # 50km radius
                        recommendations.append({
                            'workspace_id': str(ws['Id']),
                            'name': ws['Name'],
                            'type': ws['Type'],
                            'rating': float(ws['AvgRating']) if not pd.isna(ws['AvgRating']) else None,
                            'distance_km': distance,
                            'city': ws['City']
                        })
                except Exception as e:
                    print(f"⚠️ Error processing workspace {ws.get('Id', 'unknown')}: {e}")
                    continue
            
            # Sort by distance and rating
            return sorted(recommendations, 
                        key=lambda x: (x['distance_km'], 
                                      -x['rating'] if x['rating'] is not None else 0))[:top_n]
        except Exception as e:
            print(f"❌ Error in distance-based recommendations: {e}")
            return []
