# import mysql.connector
# from mysql.connector import Error
# from azure.ai.inference import ChatCompletionsClient
# from azure.ai.inference.models import SystemMessage, UserMessage
# from azure.core.credentials import AzureKeyCredential
# import json
# import time
# import traceback

# # DeepSeek configuration
# DEEPSEEK_CONFIG = {
#     'endpoint': "https://models.github.ai/inference",
#     'model': "deepseek/DeepSeek-V3-0324",
#     'token': "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"
# }

# def parse_ai_response(text_response):
#     """Parse the AI's text response into structured JSON"""
#     try:
#         # Try to find JSON in the response
#         start_idx = text_response.find('{')
#         end_idx = text_response.rfind('}') + 1
#         json_str = text_response[start_idx:end_idx]
#         return json.loads(json_str)
#     except:
#         # Fallback to manual parsing if JSON not found
#         recommendations = []
#         parts = text_response.split('\n\n')
#         for part in parts:
#             if 'Room Type:' in part:
#                 rec = {}
#                 lines = part.split('\n')
#                 for line in lines:
#                     if 'Room Type:' in line:
#                         rec['type'] = line.split(':')[1].strip()
#                     elif 'Description:' in line:
#                         rec['description'] = line.split(':')[1].strip()
#                     elif 'Capacity:' in line:
#                         rec['capacity'] = line.split(':')[1].strip()
#                     elif 'Price Range:' in line:
#                         rec['price_range'] = line.split(':')[1].strip()
#                     elif 'Benefits:' in line:
#                         rec['benefits'] = line.split(':')[1].strip()
#                 if rec:
#                     recommendations.append(rec)
#         return {'recommendations': recommendations[:3]}  # Return max 3 recommendations

# def get_fallback_recommendations():
#     """Provide fallback recommendations when AI service fails"""
#     return {
#         "recommendations": [
#             {
#                 "type": "Conference Room",
#                 "description": "Professional meeting space equipped with modern video conferencing equipment and presentation tools",
#                 "capacity": "10-15 people",
#                 "price_range": "$50-75/hour",
#                 "benefits": "Perfect for client meetings and team presentations"
#             },
#             {
#                 "type": "Creative Studio",
#                 "description": "Open space with natural lighting and flexible furniture arrangements for creative work",
#                 "capacity": "5-8 people",
#                 "price_range": "$40-60/hour",
#                 "benefits": "Ideal for brainstorming and creative collaboration"
#             },
#             {
#                 "type": "Private Office",
#                 "description": "Quiet, focused workspace with ergonomic furniture and privacy features",
#                 "capacity": "1-2 people",
#                 "price_range": "$30-45/hour",
#                 "benefits": "Suitable for concentrated individual work"
#             }
#         ]
#     }

# def get_room_recommendations(workspace_id, db_config):
#     """Get AI-powered room recommendations for a workspace"""
#     connection = None
#     cursor = None
    
#     try:
#         # Database connection
#         print(f"Connecting to database for workspace_id: {workspace_id}")
#         connection = mysql.connector.connect(**db_config)
#         cursor = connection.cursor(dictionary=True)
        
#         # 1. Get the target workspace details
#         cursor.execute("SELECT * FROM Workspace WHERE Id = %s", (workspace_id,))
#         target_workspace = cursor.fetchone()
        
#         if not target_workspace:
#             print(f"Workspace not found: {workspace_id}")
#             return {"error": "Workspace not found"}
        
#         workspace_type = target_workspace['Type']
#         workspace_name = target_workspace['Name']
        
#         # 2. Get all rooms in the target workspace
#         cursor.execute("SELECT Type, Description, Capacity FROM Room WHERE WorkspaceId = %s", (workspace_id,))
#         current_rooms = cursor.fetchall()
        
#         # 3. Get distinct room types from similar workspaces
#         cursor.execute("""
#             SELECT DISTINCT r.Type, r.Description, r.Capacity, r.PricePerHour
#             FROM Room r
#             JOIN Workspace w ON r.WorkspaceId = w.Id
#             WHERE w.Type = %s AND w.Id != %s
#             LIMIT 10
#         """, (workspace_type, workspace_id))
#         similar_rooms = cursor.fetchall()
        
#         # 4. Format the information for the prompt
#         current_rooms_str = "\n".join([
#             f"- {room['Type']} (Capacity: {room['Capacity']}): {room['Description']}"
#             for room in current_rooms
#         ]) or "No rooms currently available"
        
#         similar_rooms_str = "\n".join([
#             f"- {room['Type']} (Capacity: {room['Capacity']}, Price: ${room['PricePerHour']}/hr): {room['Description']}"
#             for room in similar_rooms
#         ]) or "No similar rooms found in other workspaces"
        
#         # 5. Construct the prompt
#         prompt = f"""
#         **Workspace Details:**
#         - Name: {workspace_name}
#         - Type: {workspace_type}
        
#         **Current Rooms:**
#         {current_rooms_str}
        
#         **Rooms in Similar Workspaces:**
#         {similar_rooms_str}
        
#         **Task:**
#         Recommend exactly 3 new room types we should add. For each recommendation, provide:
        
#         Please format your response as valid JSON with this structure:
#         {{
#             "recommendations": [
#                 {{
#                     "type": "Room Type Name",
#                     "description": "Brief description (40-60 words)",
#                     "capacity": "Ideal capacity range",
#                     "price_range": "Suggested price range",
#                     "benefits": "Key benefits for our workspace"
#                 }},
#                 // ... 2 more recommendations
#             ]
#         }}
#         """
        
#         try:
#             print("Initializing AI client")
#             # Initialize DeepSeek client
#             client = ChatCompletionsClient(
#                 endpoint=DEEPSEEK_CONFIG['endpoint'],
#                 credential=AzureKeyCredential(DEEPSEEK_CONFIG['token']),
#             )

#             print("Sending request to AI model")
#             # Get the recommendation from the model
#             response = client.complete(
#                 messages=[
#                     SystemMessage("You are a workspace design expert. Respond with exactly 3 room recommendations in perfect JSON format."),
#                     UserMessage(prompt),
#                 ],
#                 temperature=0.7,
#                 top_p=0.9,
#                 max_tokens=1024,
#                 model=DEEPSEEK_CONFIG['model']
#             )
            
#             print("Processing AI response")
#             # Parse the response
#             ai_response = response.choices[0].message.content
#             parsed_response = parse_ai_response(ai_response)
            
#         except Exception as e:
#             print(f"AI service error: {str(e)}")
#             print(f"Traceback: {traceback.format_exc()}")
#             print("Using fallback recommendations")
#             parsed_response = get_fallback_recommendations()
        
#         # Build final response
#         result = {
#             "workspace_id": workspace_id,
#             "workspace_name": workspace_name,
#             "workspace_type": workspace_type,
#             "recommendations": parsed_response.get('recommendations', [])[:3]  # Ensure only 3
#         }
        
#         print(f"Successfully generated recommendations for workspace {workspace_id}")
#         return result
        
#     except Error as e:
#         print(f"Database error: {str(e)}")
#         return {"error": f"Database error: {str(e)}"}
#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         print(f"Error type: {type(e)}")
#         print(f"Traceback: {traceback.format_exc()}")
#         return {"error": f"Unexpected error: {str(e)}"}
#     finally:
#         if cursor:
#             cursor.close()
#         if connection and connection.is_connected():
#             connection.close()
#             print("Database connection closed")




import mysql.connector
from mysql.connector import Error
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import json
import time
import sys
import os

# Import fallback recommendations
try:
    from fallback_recommendations import get_random_recommendations
    print("Successfully imported fallback recommendations")
except ImportError:
    print("Warning: Could not import fallback recommendations, using built-in fallbacks")
    # Define a simple fallback function in case the import fails
    def get_random_recommendations(count=3):
        return [
            {"type": "Standard Conference Room", "description": "A versatile meeting space for team gatherings, client presentations, and collaborative sessions.", "capacity": "8-12", "price_range": "$25-35/hr", "benefits": "Professional atmosphere for important meetings"},
            {"type": "Focus Booth", "description": "Individual workspace designed for concentrated work and privacy when needed.", "capacity": "1", "price_range": "$10-15/hr", "benefits": "Quiet space for deep work and productivity"},
            {"type": "Innovation Lab", "description": "Flexible space with modular furniture and tools for creative teamwork and ideation.", "capacity": "6-10", "price_range": "$30-40/hr", "benefits": "Promotes creative thinking and problem-solving"}
        ][:count]

# DeepSeek configuration
DEEPSEEK_CONFIG = {
    'endpoint': "https://models.github.ai/inference",
    'model': "deepseek/DeepSeek-V3-0324",
    'token': "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"
}

def parse_ai_response(text_response):
    """Parse the AI's text response into structured JSON"""
    try:
        # If response is already a dict, return it
        if isinstance(text_response, dict):
            return text_response
            
        # Try to find JSON in the response
        start_idx = text_response.find('{')
        end_idx = text_response.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > 0:
            json_str = text_response[start_idx:end_idx]
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"JSON parsing failed: {str(e)}")
        # Fallback to manual parsing if JSON not found
        recommendations = []
        parts = text_response.split('\n\n')
        for part in parts:
            if 'Room Type:' in part:
                rec = {}
                lines = part.split('\n')
                for line in lines:
                    if 'Room Type:' in line:
                        rec['type'] = line.split(':')[1].strip()
                    elif 'Description:' in line:
                        rec['description'] = line.split(':')[1].strip()
                    elif 'Capacity:' in line:
                        rec['capacity'] = line.split(':')[1].strip()
                    elif 'Price Range:' in line:
                        rec['price_range'] = line.split(':')[1].strip()
                    elif 'Benefits:' in line:
                        rec['benefits'] = line.split(':')[1].strip()
                if rec:
                    recommendations.append(rec)
        
        # If no recommendations were parsed, use random fallback recommendations
        if not recommendations:
            recommendations = get_random_recommendations(3)
        
        return {'recommendations': recommendations[:3]}  # Return max 3 recommendations

def get_room_recommendations(workspace_id, db_config):
    try:
        # Database connection
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        
        # 1. Get the target workspace details
        cursor.execute("SELECT * FROM Workspace WHERE Id = %s", (workspace_id,))
        target_workspace = cursor.fetchone()
        
        if not target_workspace:
            return {"error": "Workspace not found"}
        
        workspace_type = target_workspace['Type']
        workspace_name = target_workspace['Name']
        
        # 2. Get all rooms in the target workspace
        cursor.execute("SELECT Type, Description, Capacity FROM Room WHERE WorkspaceId = %s", (workspace_id,))
        current_rooms = cursor.fetchall()
        
        # 3. Get distinct room types from similar workspaces
        cursor.execute("""
            SELECT DISTINCT r.Type, r.Description, r.Capacity, r.PricePerHour
            FROM Room r
            JOIN Workspace w ON r.WorkspaceId = w.Id
            WHERE w.Type = %s AND w.Id != %s
            LIMIT 10
        """, (workspace_type, workspace_id))
        similar_rooms = cursor.fetchall()
        
        # 4. Format the information for the prompt
        current_rooms_str = "\n".join([
            f"- {room['Type']} (Capacity: {room['Capacity']}): {room['Description']}"
            for room in current_rooms
        ]) or "No rooms currently available"
        
        similar_rooms_str = "\n".join([
            f"- {room['Type']} (Capacity: {room['Capacity']}, Price: ${room['PricePerHour']}/hr): {room['Description']}"
            for room in similar_rooms
        ]) or "No similar rooms found in other workspaces"
        
        # 5. Construct the prompt with explicit JSON formatting instructions
        prompt = f"""
        **Workspace Details:**
        - Name: {workspace_name}
        - Type: {workspace_type}
        
        **Current Rooms:**
        {current_rooms_str}
        
        **Rooms in Similar Workspaces:**
        {similar_rooms_str}
        
        **Task:**
        Recommend exactly 3 new room types we should add. For each recommendation, provide:
        
        Please format your response as valid JSON with this structure:
        {{
            "recommendations": [
                {{
                    "type": "Room Type Name",
                    "description": "Brief description (40-60 words)",
                    "capacity": "Ideal capacity range",
                    "price_range": "Suggested price range",
                    "benefits": "Key benefits for our workspace"
                }},
                // ... 2 more recommendations
            ]
        }}
        
        Focus on practical, valuable additions that differentiate our workspace.
        """
        
        try:
            # Initialize DeepSeek client
            client = ChatCompletionsClient(
                endpoint=DEEPSEEK_CONFIG['endpoint'],
                credential=AzureKeyCredential(DEEPSEEK_CONFIG['token']),
            )

            # Get the recommendation from the model
            response = client.complete(
                messages=[
                    SystemMessage("You are a workspace design expert. Respond with exactly 3 room recommendations in perfect JSON format."),
                    UserMessage(prompt),
                ],
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
                model=DEEPSEEK_CONFIG['model']
            )

            # Add debug logging
            print("Raw AI response type:", type(response))
            
            # Process the response with robust error handling
            try:
                # Check if the response has the expected structure
                if hasattr(response, 'choices') and response.choices:
                    ai_response = response.choices[0].message.content
                    print(f"AI response type: {type(ai_response)}")
                    print(f"AI response content preview: {ai_response[:200]}...") # Print first 200 chars
                    
                    parsed_response = parse_ai_response(ai_response)
                else:
                    # Alternative parsing if response structure is different
                    print(f"Unexpected response structure: {response}")
                    if isinstance(response, str):
                        parsed_response = parse_ai_response(response)
                    else:
                        # Try to convert to string if it's something else
                        parsed_response = parse_ai_response(str(response))
            except Exception as e:
                print(f"Error parsing AI response: {str(e)}")
                print(f"Response was: {response}")
                # Use random fallback recommendations
                parsed_response = {"recommendations": get_random_recommendations(3)}
            
            print("Final parsed response:", parsed_response)
            
        except Exception as ai_error:
            print(f"AI service error: {str(ai_error)}")
            # Use random fallback recommendations if the AI service fails
            parsed_response = {"recommendations": get_random_recommendations(3)}
        
        # Build final response
        result = {
            "workspace_id": workspace_id,
            "workspace_name": workspace_name,
            "workspace_type": workspace_type,
            "recommendations": parsed_response.get('recommendations', [])[:3]  # Ensure only 3
        }
        
        return result
        
    except Error as e:
        print(f"Database error: {str(e)}")
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        print(f"General error: {str(e)}")
        return {"error": f"AI model error: {str(e)}"}
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()