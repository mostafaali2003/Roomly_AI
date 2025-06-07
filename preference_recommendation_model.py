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

class PreferenceWorkspaceRecommender:
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
            SELECT 
                w.Id, w.Name, w.Type, w.AvgRating, w.Description,
                ANY_VALUE(l.Latitude) as Latitude, 
                ANY_VALUE(l.Longitude) as Longitude, 
                ANY_VALUE(l.City) as City, 
                ANY_VALUE(l.Town) as Town,
                MIN(r.PricePerHour) as MinPrice,
                MAX(r.PricePerHour) as MaxPrice,
                GROUP_CONCAT(DISTINCT a.Name) as Amenities
            FROM Workspace w
            JOIN Location l ON w.LocationId = l.Id
            JOIN Room r ON r.WorkspaceId = w.Id
            LEFT JOIN Amenity a ON a.RoomId = r.Id
            GROUP BY w.Id, w.Name, w.Type, w.AvgRating, w.Description
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
    
    def get_top_rated_workspaces(self, top_n: int) -> List[Dict]:
        """Get top rated workspaces with complete information"""
        query = """
        SELECT 
            w.Id, w.Name, w.Type, w.AvgRating, w.Description,
            ANY_VALUE(l.City) as City, ANY_VALUE(l.Town) as Town,
            MIN(r.PricePerHour) as MinPrice,
            MAX(r.PricePerHour) as MaxPrice,
            GROUP_CONCAT(DISTINCT a.Name) as Amenities
        FROM Workspace w
        JOIN Location l ON w.LocationId = l.Id
        JOIN Room r ON r.WorkspaceId = w.Id
        LEFT JOIN Amenity a ON a.RoomId = r.Id
        GROUP BY w.Id, w.Name, w.Type, w.AvgRating, w.Description
        ORDER BY w.AvgRating DESC
        LIMIT %s
        """
        try:
            workspaces = pd.read_sql(query, self.connection, params=(top_n,))
            return [{
                'workspace_id': row['Id'],
                'name': row['Name'],
                'type': row['Type'],
                'rating': float(row['AvgRating']) if not pd.isna(row['AvgRating']) else None,
                'city': row['City'],
                'town': row['Town'],
                'price_range': f"${row['MinPrice']}-${row['MaxPrice']}/hr",
                'amenities': row['Amenities'].split(',') if row['Amenities'] else [],
                'description': (row['Description'][:100] + '...') if row['Description'] else '',
                'match_status': 'top rated'
            } for _, row in workspaces.iterrows()]
        except Exception as e:
            print(f"⚠️ Error getting top rated workspaces: {e}")
            return []

    def get_preference_based_recommendations(self, user_id: str, top_n: int = 5) -> List[Dict]:
        """
        Recommend workspaces based on user preferences from the Preference table
        Returns top N workspaces matching the user's preferences
        """
        print(f"\n✨ Generating preference-based recommendations for user {user_id}...")
        
        try:
            # 1. Load user preferences
            if not self.connect():
                print("❌ Database connection failed")
                return []
                
            # Get user preferences
            pref_query = f"""
            SELECT BudgetPreference, WorkspaceTypePreference 
            FROM Preference 
            WHERE UserId = '{user_id}'
            """
            preferences = pd.read_sql(pref_query, self.connection)
            
            if preferences.empty:
                print("ℹ️ No preferences found for user - returning top rated workspaces")
                return self.get_top_rated_workspaces(top_n)
                
            budget_pref = preferences['BudgetPreference'].iloc[0]
            type_pref = preferences['WorkspaceTypePreference'].iloc[0]
            
            print(f"👤 User Preferences - Budget: {budget_pref}, Workspace Type: {type_pref}")

            # 2. Load all workspaces with their amenities and pricing
            workspace_query = """
            SELECT 
                w.Id, w.Name, w.Type, w.AvgRating, w.Description,
                ANY_VALUE(l.City) as City, 
                ANY_VALUE(l.Town) as Town,
                MIN(r.PricePerHour) as MinPrice,
                MAX(r.PricePerHour) as MaxPrice,
                GROUP_CONCAT(DISTINCT a.Name) as Amenities
            FROM Workspace w
            JOIN Location l ON w.LocationId = l.Id
            JOIN Room r ON r.WorkspaceId = w.Id
            LEFT JOIN Amenity a ON a.RoomId = r.Id
            GROUP BY w.Id, w.Name, w.Type, w.AvgRating, w.Description
            """
            workspaces = pd.read_sql(workspace_query, self.connection)
            
            if workspaces.empty:
                print("❌ No workspaces found in database")
                return []

            # 3. Apply preference filters
            filtered_workspaces = workspaces.copy()
            
            # Filter by workspace type if specified
            if type_pref and str(type_pref).lower() != 'any':
                filtered_workspaces = filtered_workspaces[
                    filtered_workspaces['Type'].str.lower() == str(type_pref).lower()
                ]
                print(f"🔍 Filtered by type '{type_pref}': {len(filtered_workspaces)} matches")
            
            # Filter by budget if specified
            if budget_pref and str(budget_pref).lower() != 'any':
                budget_ranges = {
                    'low': (0, 50),
                    'medium': (50, 100),
                    'high': (100, float('inf'))
                }
                
                budget_key = str(budget_pref).lower()
                if budget_key in budget_ranges:
                    min_b, max_b = budget_ranges[budget_key]
                    filtered_workspaces = filtered_workspaces[
                        (filtered_workspaces['MinPrice'] >= min_b) & 
                        (filtered_workspaces['MaxPrice'] < max_b)
                    ]
                    print(f"💰 Filtered by budget '{budget_pref}': {len(filtered_workspaces)} matches")

            # 4. If no matches after filtering, return popular alternatives
            if filtered_workspaces.empty:
                print("ℹ️ No workspaces match all preferences - showing alternatives")
                # Get workspaces of similar type or any type if none specified
                fallback_workspaces = workspaces.copy()
                if type_pref and str(type_pref).lower() != 'any':
                    similar_types = self.get_similar_workspace_types(type_pref)
                    fallback_workspaces = fallback_workspaces[
                        fallback_workspaces['Type'].str.lower().isin(similar_types)
                    ]
                
                filtered_workspaces = fallback_workspaces

            # 5. Score and rank results
            def calculate_score(row):
                """Calculate composite score based on rating and popularity"""
                rating_score = row['AvgRating'] or 0
                # Additional factors could be added here (amenities, etc.)
                return rating_score
            
            filtered_workspaces['score'] = filtered_workspaces.apply(calculate_score, axis=1)
            
            recommendations = filtered_workspaces.sort_values(
                ['score', 'AvgRating'], 
                ascending=[False, False]
            ).head(top_n)

            # 6. Format results
            result = []
            for _, row in recommendations.iterrows():
                result.append({
                    'workspace_id': row['Id'],
                    'name': row['Name'],
                    'type': row['Type'],
                    'rating': float(row['AvgRating']) if not pd.isna(row['AvgRating']) else None,
                    'city': row['City'],
                    'town': row['Town'],
                    'price_range': f"${row['MinPrice']}-${row['MaxPrice']}/hr",
                    'amenities': row['Amenities'].split(',') if row['Amenities'] else [],
                    'description': (row['Description'][:100] + '...') if row['Description'] else '',
                    'match_status': self.get_match_status(row, budget_pref, type_pref)
                })
            
            print(f"✅ Found {len(result)} recommendations")
            return result
            
        except Exception as e:
            print(f"❌ Error in preference-based recommendation: {e}")
            return self.get_top_rated_workspaces(top_n)
        finally:
            self.disconnect()

    def get_similar_workspace_types(self, preferred_type: str) -> List[str]:
        """Return similar workspace types when exact matches aren't available"""
        type_mapping = {
            'coworking': ['shared office', 'flex workspace'],
            'private office': ['dedicated desk', 'executive suite'],
            'meeting room': ['conference room', 'boardroom'],
            'event space': ['conference center', 'party venue']
        }
        preferred_type = preferred_type.lower()
        return type_mapping.get(preferred_type, [preferred_type])

    def get_match_status(self, row, budget_pref: str, type_pref: str) -> str:
        """Determine how well the workspace matches preferences"""
        matches = []
        
        if type_pref and str(type_pref).lower() != 'any':
            if str(row['Type']).lower() == str(type_pref).lower():
                matches.append("type")
            elif str(type_pref).lower() in self.get_similar_workspace_types(row['Type']):
                matches.append("similar type")
        
        if budget_pref and str(budget_pref).lower() != 'any':
            budget_ranges = {
                'low': (0, 50),
                'medium': (50, 100),
                'high': (100, float('inf'))
            }
            budget_key = str(budget_pref).lower()
            if budget_key in budget_ranges:
                min_b, max_b = budget_ranges[budget_key]
                if (row['MinPrice'] >= min_b) and (row['MaxPrice'] < max_b):
                    matches.append("budget")
        
        if not matches:
            return "popular alternative"
        elif len(matches) == 2:
            return "perfect match"
        else:
            return f"matches {matches[0]}"
