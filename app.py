from flask import Flask, request, jsonify
import os
import sys
import datetime
from mysql.connector import Error
import warnings
import pandas as pd

# Add simple logging via print
print(f"=== STARTING APPLICATION AT {datetime.datetime.now().isoformat()} ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import all models
try:
    print("Attempting to import models...")
    from nearby_recommendation_model import NearbyWorkspaceRecommender
    from cancellation_model import CancellationFeeModel
    from workspace_analysis_model import WorkspaceDashboard
    from staff_room_recommendation_model import get_room_recommendations
    from rate_recommendation_model import RateWorkspaceRecommender
    from preference_recommendation_model import PreferenceWorkspaceRecommender
    print("All models successfully imported")
except Exception as e:
    print(f"ERROR IMPORTING MODELS: {str(e)}")
    print(f"Error type: {type(e)}")
    print(f"Error traceback: {sys.exc_info()}")
    raise

# Suppress warnings 
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Database configuration
db_config = {
    'host': 'gateway01.us-west-2.prod.aws.tidbcloud.com',
    'database': 'Roomly',
    'user': '3hCuA6E2KTDN8gC.root',
    'password': 'etj31aRtKnWBaFGa',
    'port': 4000,
}

print("Initializing models...")

try:
    # Initialize models
    nearby_recommender = NearbyWorkspaceRecommender(db_config)
    print("NearbyWorkspaceRecommender initialized")
    
    cancellation_model = CancellationFeeModel(db_config)
    print("CancellationFeeModel initialized")
    
    workspace_dashboard = WorkspaceDashboard(
        host=db_config['host'],
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )
    print("WorkspaceDashboard initialized")
    
    rate_recommender = RateWorkspaceRecommender(db_config)
    print("RateWorkspaceRecommender initialized")
    
    preference_recommender = PreferenceWorkspaceRecommender(db_config)
    print("PreferenceWorkspaceRecommender initialized")
    print("All models initialized successfully")
except Exception as e:
    print(f"ERROR INITIALIZING MODELS: {str(e)}")
    print(f"Error type: {type(e)}")
    print(f"Error traceback: {sys.exc_info()}")

# Helper function for standardized responses
def create_response(status, data=None, error=None):
    """Create a standardized API response"""
    response = {
        "status": status,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    if data is not None:
        response["data"] = data
    
    if error is not None:
        response["error"] = error
    
    return response

# API Routes
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    print(f"Health check endpoint called at {datetime.datetime.now().isoformat()}")
    return jsonify(create_response("success", {"message": "API is healthy"}))

# 1. Nearby Workspace Recommendations
@app.route('/api/v1/recommendations/nearby', methods=['GET'])
def nearby_recommendations():
    """Get nearby workspace recommendations"""
    print(f"Nearby recommendations endpoint called with params: {request.args}")
    try:
        # Get parameters
        user_id = request.args.get('user_id')
        latitude = request.args.get('latitude')
        longitude = request.args.get('longitude')
        top_n = request.args.get('top_n', default=5, type=int)
        
        print(f"Processing nearby recommendations for user_id={user_id}, lat={latitude}, lon={longitude}, top_n={top_n}")
        
        # Validate required parameters
        if not user_id or not latitude or not longitude:
            print("Missing required parameters")
            return jsonify(create_response("error", error="Missing required parameters: user_id, latitude, longitude")), 400
        
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except ValueError:
            print("Invalid coordinates format")
            return jsonify(create_response("error", error="Invalid coordinates format")), 400
        
        # Get recommendations
        print("Calling nearby_recommender.recommend_workspaces")
        recommendations = nearby_recommender.recommend_workspaces(
            user_id=user_id,
            user_lat=latitude,
            user_lon=longitude,
            top_n=top_n
        )
        print(f"Got {len(recommendations)} recommendations")
        
        return jsonify(create_response("success", {
            "recommendations": recommendations,
            "count": len(recommendations)
        }))
        
    except Exception as e:
        print(f"ERROR in nearby_recommendations: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {sys.exc_info()}")
        return jsonify(create_response("error", error=str(e))), 500

# 2. Preference-Based Recommendations
@app.route('/api/v1/recommendations/preferences', methods=['GET'])
def preference_recommendations():
    """Get preference-based workspace recommendations"""
    print(f"Preference recommendations endpoint called with params: {request.args}")
    try:
        # Get parameters
        user_id = request.args.get('user_id')
        top_n = request.args.get('top_n', default=5, type=int)
        
        print(f"Processing preference recommendations for user_id={user_id}, top_n={top_n}")
        
        # Validate required parameters
        if not user_id:
            print("Missing required parameter: user_id")
            return jsonify(create_response("error", error="Missing required parameter: user_id")), 400
        
        # Get recommendations
        print("Calling preference_recommender.get_preference_based_recommendations")
        recommendations = preference_recommender.get_preference_based_recommendations(
            user_id=user_id,
            top_n=top_n
        )
        print(f"Got {len(recommendations)} recommendations")
        
        return jsonify(create_response("success", {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }))
        
    except Exception as e:
        print(f"ERROR in preference_recommendations: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {sys.exc_info()}")
        return jsonify(create_response("error", error=str(e))), 500

# 3. Rating-Based Recommendations
@app.route('/api/v1/recommendations/ratings', methods=['GET'])
def rating_recommendations():
    """Get rating-based workspace recommendations"""
    print(f"Rating recommendations endpoint called with params: {request.args}")
    try:
        # Get parameters
        user_id = request.args.get('user_id')
        top_n = request.args.get('top_n', default=5, type=int)
        
        print(f"Processing rating recommendations for user_id={user_id}, top_n={top_n}")
        
        # Validate required parameters
        if not user_id:
            print("Missing required parameter: user_id")
            return jsonify(create_response("error", error="Missing required parameter: user_id")), 400
        
        # Get recommendations
        print("Calling rate_recommender.recommend_workspaces")
        recommendations = rate_recommender.recommend_workspaces(
            user_id=user_id,
            top_n=top_n
        )
        print(f"Got {len(recommendations)} recommendations")
        
        return jsonify(create_response("success", {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }))
        
    except Exception as e:
        print(f"ERROR in rating_recommendations: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {sys.exc_info()}")
        return jsonify(create_response("error", error=str(e))), 500

# 4. Staff Room Recommendations
# @app.route('/api/v1/recommendations/rooms', methods=['GET'])
# def room_recommendations():
#     """Get AI-generated room recommendations for a workspace"""
#     print(f"Room recommendations endpoint called with params: {request.args}")
#     try:
#         # Get parameters
#         workspace_id = request.args.get('workspace_id')
        
#         print(f"Processing room recommendations for workspace_id={workspace_id}")
        
#         # Validate required parameters
#         if not workspace_id:
#             print("Missing required parameter: workspace_id")
#             return jsonify(create_response("error", error="Missing required parameter: workspace_id")), 400
        
#         # Get recommendations
#         print("Calling get_room_recommendations")
#         recommendations = get_room_recommendations(workspace_id, db_config)
        
#         if "error" in recommendations:
#             print(f"Error in get_room_recommendations: {recommendations['error']}")
#             return jsonify(create_response("error", error=recommendations["error"])), 500
        
#         print("Room recommendations successful")
#         return jsonify(create_response("success", recommendations))
        
#     except Exception as e:
#         print(f"ERROR in room_recommendations: {str(e)}")
#         print(f"Error type: {type(e)}")
#         print(f"Error traceback: {sys.exc_info()}")
#         return jsonify(create_response("error", error=str(e))), 500

@app.route('/api/v1/recommendations/rooms', methods=['GET'])
def room_recommendations():
    """Get AI-generated room recommendations for a workspace"""
    print(f"Room recommendations endpoint called with params: {request.args}")
    try:
        # Get parameters
        workspace_id = request.args.get('workspace_id')
        
        print(f"Processing room recommendations for workspace_id={workspace_id}")
        
        # Validate required parameters
        if not workspace_id:
            print("Missing required parameter: workspace_id")
            return jsonify(create_response("error", error="Missing required parameter: workspace_id")), 400
        
        # Get recommendations
        print("Calling get_room_recommendations")
        recommendations = get_room_recommendations(workspace_id, db_config)
        
        if "error" in recommendations:
            print(f"Error in get_room_recommendations: {recommendations['error']}")
            return jsonify(create_response("error", error=recommendations["error"])), 500
        
        print("Room recommendations successful")
        return jsonify(create_response("success", recommendations))
        
    except Exception as e:
        print(f"ERROR in room_recommendations: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {sys.exc_info()}")
        return jsonify(create_response("error", error=str(e))), 500

# 5. Cancellation Fee Recommendations (All)
@app.route('/api/v1/recommendations/cancellation-fees', methods=['GET'])
def cancellation_fee_recommendations():
    """Get cancellation fee recommendations for all workspaces"""
    print("Cancellation fee recommendations endpoint called")
    try:
        # Get recommendations
        print("Calling cancellation_model.generate_recommendations")
        recommendations = cancellation_model.generate_recommendations()
        
        if recommendations is None:
            print("Failed to generate cancellation fee recommendations")
            return jsonify(create_response("error", error="Failed to generate recommendations")), 500
        
        print(f"Got {len(recommendations)} cancellation fee recommendations")
        return jsonify(create_response("success", {
            "recommendations": recommendations.to_dict(orient='records')
        }))
        
    except Exception as e:
        print(f"ERROR in cancellation_fee_recommendations: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {sys.exc_info()}")
        return jsonify(create_response("error", error=str(e))), 500

# 6. Workspace-Specific Cancellation Fee Recommendations
@app.route('/api/v1/recommendations/cancellation-fees/<workspace_id>', methods=['GET'])
def workspace_cancellation_fee_recommendations(workspace_id):
    """Get cancellation fee recommendations for a specific workspace"""
    print(f"Workspace-specific cancellation fee recommendations endpoint called for workspace_id={workspace_id}")
    try:
        # Get recommendations
        print(f"Calling cancellation_model.get_workspace_recommendations for workspace_id={workspace_id}")
        recommendations = cancellation_model.get_workspace_recommendations(workspace_id)
        
        if recommendations is None:
            print(f"No recommendations found for workspace {workspace_id}")
            return jsonify(create_response("error", error=f"No recommendations found for workspace {workspace_id}")), 404
        
        print(f"Got {len(recommendations)} cancellation fee recommendations for workspace {workspace_id}")
        return jsonify(create_response("success", {
            "workspace_id": workspace_id,
            "recommendations": recommendations.to_dict(orient='records')
        }))
        
    except Exception as e:
        print(f"ERROR in workspace_cancellation_fee_recommendations: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {sys.exc_info()}")
        return jsonify(create_response("error", error=str(e))), 500

# 7. Workspace Analysis
@app.route('/api/v1/analysis/workspace/<string:workspace_id>', methods=['GET'])
def workspace_analysis(workspace_id):
    """Get comprehensive analysis for a workspace"""
    print(f"Workspace analysis endpoint called for workspace_id={workspace_id}")
    try:
        # Get analysis
        print(f"Calling workspace_dashboard.get_all_statistics for workspace_id={workspace_id}")
        analysis = workspace_dashboard.get_all_statistics(workspace_id)
        
        if "error" in analysis:
            print(f"Error in workspace_dashboard.get_all_statistics: {analysis['error']}")
            return jsonify(create_response("error", error=analysis["error"])), 500
        
        print(f"Successfully retrieved analysis for workspace {workspace_id}")
        return jsonify(create_response("success", {
            "workspace_id": workspace_id,
            "analysis": analysis
        }))
        
    except Exception as e:
        print(f"ERROR in workspace_analysis: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {sys.exc_info()}")
        return jsonify(create_response("error", error=str(e))), 500

# Model Training Endpoint (Optional)
@app.route('/api/v1/models/train/ratings', methods=['POST'])
def train_rating_model():
    """Train the rating-based recommendation model"""
    print("Train rating model endpoint called")
    try:
        # Train model
        print("Calling rate_recommender.build_recommendation_model")
        rate_recommender.build_recommendation_model()
        
        print("Rating model trained successfully")
        return jsonify(create_response("success", {
            "message": "Rating model trained successfully"
        }))
        
    except Exception as e:
        print(f"ERROR in train_rating_model: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback: {sys.exc_info()}")
        return jsonify(create_response("error", error=str(e))), 500

# Add a port configuration that respects Railway's PORT environment variable
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting Flask app on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)
else:
    print("Running in production mode via Gunicorn")
