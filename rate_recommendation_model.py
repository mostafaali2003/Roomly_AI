import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import warnings
from typing import Dict, List

# Suppress warnings
warnings.filterwarnings('ignore')

class RateWorkspaceRecommender:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection = None
        self.model = None
        self.index = None
        self.reviews_df = None
        self.users_df = None
        self.workspaces_df = None
        self.user_lookup = None
        self.workspace_lookup = None
        
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            return True
        except Exception as e:
            print(f"❌ MySQL connection error: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
    
    def load_data(self) -> bool:
        """Load and preprocess all required data"""
        if not self.connect():
            return False
            
        try:
            print("⏳ Loading data from database...")
            
            # Fetch reviews data
            reviews_query = """
                SELECT r.Id as review_id, r.UserId as user_id, r.WorkspaceId as workspace_id, 
                       r.Rating, r.Comment, r.ReviewDate
                FROM Review r
            """
            self.reviews_df = pd.read_sql(reviews_query, self.connection)
            
            # Fetch user data
            users_query = "SELECT Id as user_id, FName, LName, Email FROM User"
            self.users_df = pd.read_sql(users_query, self.connection)
            
            # Fetch workspace data with additional details
            workspaces_query = """
                SELECT 
                    w.Id as workspace_id, 
                    w.Name, 
                    w.Description, 
                    w.AvgRating, 
                    w.Type,
                    ANY_VALUE(l.City) as City,
                    ANY_VALUE(l.Town) as Town,
                    MIN(rm.PricePerHour) as min_price,
                    MAX(rm.PricePerHour) as max_price,
                    GROUP_CONCAT(DISTINCT a.Name) as amenities
                FROM Workspace w
                JOIN Location l ON w.LocationId = l.Id
                JOIN Room rm ON rm.WorkspaceId = w.Id
                LEFT JOIN Amenity a ON a.RoomId = rm.Id
                GROUP BY w.Id, w.Name, w.Description, w.AvgRating, w.Type
            """
            self.workspaces_df = pd.read_sql(workspaces_query, self.connection)
            
            # Preprocess data
            self.reviews_df['Rating'] = self.reviews_df['Rating'].astype(float)
            self.workspaces_df['AvgRating'] = self.workspaces_df['AvgRating'].astype(float)
            
            # Apply sentiment analysis
            self._analyze_review_sentiment()
            
            print(f"✅ Loaded {len(self.reviews_df)} reviews, {len(self.users_df)} users, and {len(self.workspaces_df)} workspaces")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        finally:
            self.disconnect()
    
    def _analyze_review_sentiment(self):
        """Analyze sentiment of reviews and create combined score"""
        def simple_sentiment(text):
            if pd.isna(text):
                return 0.5
            positive_words = ['great', 'excellent', 'amazing', 'professional', 'helpful', 
                            'comfortable', 'good', 'nice', 'beautiful', 'luxury', 'unique']
            negative_words = ['could be', 'but', 'limited', 'crowded', 'issue']
            
            text_lower = text.lower()
            positive_score = sum(1 for word in positive_words if word in text_lower)
            negative_score = sum(1 for word in negative_words if word in text_lower)
            
            if positive_score > negative_score:
                return 1.0
            elif negative_score > positive_score:
                return 0.0
            else:
                return 0.5
        
        self.reviews_df['Sentiment'] = self.reviews_df['Comment'].apply(simple_sentiment)
        self.reviews_df['CombinedScore'] = (self.reviews_df['Rating']/5.0 * 0.7 + self.reviews_df['Sentiment'] * 0.3)
    
    def prepare_tf_data(self):
        """Prepare TensorFlow datasets from loaded data"""
        print("⏳ Preparing TensorFlow datasets...")
        
        # Create StringLookup layers
        self.user_lookup = tf.keras.layers.StringLookup()
        self.user_lookup.adapt(self.reviews_df['user_id'].unique())
        
        self.workspace_lookup = tf.keras.layers.StringLookup()
        self.workspace_lookup.adapt(self.reviews_df['workspace_id'].unique())
        
        # Convert to TensorFlow datasets
        ratings_dataset = tf.data.Dataset.from_tensor_slices({
            'user_id': self.reviews_df['user_id'].values,
            'workspace_id': self.reviews_df['workspace_id'].values,
            'rating': self.reviews_df['CombinedScore'].values
        })
        
        # Split into train and test
        shuffled = ratings_dataset.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
        train = shuffled.take(int(0.8 * len(shuffled)))
        test = shuffled.skip(int(0.8 * len(shuffled))).take(int(0.2 * len(shuffled)))
        
        return train, test
    
    def build_recommendation_model(self):
        """Build and train the TensorFlow recommendation model"""
        if self.reviews_df is None:
            if not self.load_data():
                raise Exception("Failed to load data for model building")
        
        train, test = self.prepare_tf_data()
        
        print("⏳ Building recommendation model...")
        
        # Define the model
        class RecommendationModel(tfrs.Model):
            def __init__(self, user_lookup, workspace_lookup, embedding_dim=32):
                super().__init__()
                self.user_lookup = user_lookup
                self.workspace_lookup = workspace_lookup
                
                # User and workspace embedding models
                self.user_embedding = tf.keras.Sequential([
                    user_lookup,
                    tf.keras.layers.Embedding(user_lookup.vocabulary_size(), embedding_dim),
                    tf.keras.layers.Dense(embedding_dim, activation='relu')
                ])
                
                self.workspace_embedding = tf.keras.Sequential([
                    workspace_lookup,
                    tf.keras.layers.Embedding(workspace_lookup.vocabulary_size(), embedding_dim),
                    tf.keras.layers.Dense(embedding_dim, activation='relu')
                ])
                
                # Rating prediction task
                self.rating_task = tfrs.tasks.Ranking(
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.RootMeanSquaredError()]
                )
            
            def call(self, features):
                user_embeddings = self.user_embedding(features['user_id'])
                workspace_embeddings = self.workspace_embedding(features['workspace_id'])
                return (
                    user_embeddings,
                    workspace_embeddings,
                    tf.reduce_sum(user_embeddings * workspace_embeddings, axis=1)
                )
            
            def compute_loss(self, features, training=False):
                user_embeddings, workspace_embeddings, predicted_ratings = self(features)
                return self.rating_task(
                    labels=features['rating'],
                    predictions=predicted_ratings
                )
        
        # Create and train model
        self.model = RecommendationModel(self.user_lookup, self.workspace_lookup)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        cached_train = train.shuffle(100_000).batch(8192).cache()
        cached_test = test.batch(4096).cache()
        
        print("⏳ Training model...")
        self.model.fit(cached_train, epochs=10, validation_data=cached_test)
        
        # Create retrieval index
        print("⏳ Building recommendation index...")
        self.index = tfrs.layers.factorized_top_k.BruteForce(self.model.user_embedding)
        self.index.index_from_dataset(
            tf.data.Dataset.from_tensor_slices(self.workspaces_df['workspace_id'].values)
            .batch(100)
            .map(lambda workspace_id: self.model.workspace_embedding(workspace_id))
        )
        
        print("✅ Model training complete")
    
    def recommend_workspaces(self, user_id: str, top_n: int = 5) -> List[Dict]:
        """Generate workspace recommendations for a user"""
        print(f"\n✨ Generating recommendations for user {user_id}...")
        
        try:
            # Load data if not loaded
            if self.reviews_df is None:
                if not self.load_data():
                    return []
            
            # Build model if not built
            if self.model is None:
                self.build_recommendation_model()
            
            # Get recommendations
            try:
                _, workspaces = self.index({
                    'user_id': np.array([user_id]),
                    'workspace_id': np.array([0])  # Dummy value
                })
                
                # Get top N recommendations
                top_workspace_ids = workspaces[0, :top_n].numpy()
                
                # Check if user has previous reviews
                user_reviews = self.reviews_df[self.reviews_df['user_id'] == user_id]
                if not user_reviews.empty:
                    match_reason = "based on your previous reviews and ratings"
                else:
                    match_reason = "based on similar users"
                
            except:
                # If any error occurs (e.g., new user), recommend top-rated workspaces
                print("⚠️ Using fallback recommendation method")
                top_workspace_ids = self.workspaces_df.sort_values('AvgRating', ascending=False)['workspace_id'].head(top_n).values
                match_reason = "top-rated workspaces"
            
            # Get workspace details for recommendations
            recommendations = []
            for ws_id in top_workspace_ids:
                try:
                    ws_data = self.workspaces_df[self.workspaces_df['workspace_id'] == ws_id].iloc[0]
                    recommendations.append({
                        'workspace_id': ws_id,
                        'name': ws_data['Name'],
                        'type': ws_data['Type'],
                        'avg_rating': float(ws_data['AvgRating']),
                        'city': ws_data['City'],
                        'town': ws_data['Town'],
                        'price_range': f"${ws_data['min_price']}-${ws_data['max_price']}/hr",
                        'amenities': ws_data['amenities'].split(',') if pd.notna(ws_data['amenities']) else [],
                        'match_reason': match_reason
                    })
                except Exception as e:
                    print(f"⚠️ Error processing workspace {ws_id}: {e}")
                    continue
            
            print(f"✅ Found {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return []
