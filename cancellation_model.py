import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, time
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class CancellationFeeModel:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
    def connect_to_db(self):
        """Establish database connection"""
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            self.cursor = self.conn.cursor(dictionary=True)
            print("Successfully connected to database")
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")
            raise

    def close_connection(self):
        """Close database connection"""
        if self.conn and self.conn.is_connected():
            self.cursor.close()
            self.conn.close()
            print("Database connection closed")

    def get_room_booking_data(self, workspace_id=None):
        """Fetch room booking data with relevant information"""
        # First check if CancellationTime column exists
        try:
            self.cursor.execute("SHOW COLUMNS FROM Reservation LIKE 'CancellationTime'")
            has_cancellation_time = bool(self.cursor.fetchone())
        except mysql.connector.Error as err:
            print(f"Error checking for CancellationTime column: {err}")
            has_cancellation_time = False

        query = """
        SELECT 
            r.Id as room_id,
            r.Name as room_name,
            r.Type as room_type,
            r.PricePerHour,
            res.StartTime,
            res.EndTime,
            res.Status,
            res.TotalCost,
            """
        
        if has_cancellation_time:
            query += "res.CancellationTime,"
        else:
            query += "NULL as CancellationTime,"
        
        query += """
            w.Name as workspace_name,
            w.Type as workspace_type,
            w.Id as workspace_id
        FROM Room r
        JOIN Booking b ON r.Id = b.RoomId
        JOIN Reservation res ON b.ReservationId = res.Id
        JOIN Workspace w ON r.WorkspaceId = w.Id
        WHERE res.Status IN ('Completed', 'Cancelled')
        """
        
        if workspace_id:
            query += f" AND w.Id = '{workspace_id}'"
        
        self.cursor.execute(query)
        return pd.DataFrame(self.cursor.fetchall())
    
    def analyze_booking_patterns(self, df):
        """Analyze booking patterns and calculate metrics"""
        if df.empty:
            print("No booking data found for the specified criteria")
            return None
            
        # Convert datetime columns
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        df['EndTime'] = pd.to_datetime(df['EndTime'])
        
        # Only try to convert CancellationTime if it exists and has data
        if 'CancellationTime' in df.columns and not df['CancellationTime'].isnull().all():
            df['CancellationTime'] = pd.to_datetime(df['CancellationTime'])
        else:
            df['CancellationTime'] = pd.NaT
        
        # Calculate booking duration in hours
        df['Duration'] = (df['EndTime'] - df['StartTime']).dt.total_seconds() / 3600
        
        # Calculate cancellation lead time (hours before booking start)
        df['CancellationLeadTime'] = np.where(
            (df['Status'] == 'Cancelled') & (~df['CancellationTime'].isna()),
            (df['StartTime'] - df['CancellationTime']).dt.total_seconds() / 3600,
            np.nan
        )
        
        # Determine peak hours (9 AM to 5 PM)
        df['IsPeakHour'] = df['StartTime'].dt.time.between(time(9), time(17))
        
        # Calculate room metrics
        room_metrics = df.groupby('room_id').agg({
            'room_name': 'first',
            'room_type': 'first',
            'PricePerHour': 'first',
            'workspace_name': 'first',
            'workspace_id': 'first',
            'Status': lambda x: (x == 'Completed').mean(),  # Completion rate
            'Duration': 'mean',  # Average booking duration
            'TotalCost': 'mean',  # Average booking cost
            'IsPeakHour': 'mean',  # Peak hour usage ratio
            'CancellationLeadTime': 'mean'  # Average cancellation lead time
        }).reset_index()
        
        # Rename columns to match the metrics we want to normalize
        room_metrics = room_metrics.rename(columns={
            'Status': 'CompletionRate',
            'Duration': 'Duration',
            'TotalCost': 'TotalCost',
            'IsPeakHour': 'PeakHourRatio'
        })
        
        # Calculate cancellation rate
        cancellation_rate = df[df['Status'] == 'Cancelled'].groupby('room_id').size() / df.groupby('room_id').size()
        room_metrics['CancellationRate'] = room_metrics['room_id'].map(cancellation_rate).fillna(0)
        
        return room_metrics

    def calculate_cancellation_fees(self, room_metrics):
        """Calculate recommended cancellation fees based on room metrics"""
        if room_metrics is None or room_metrics.empty:
            return None
            
        # Normalize metrics for scoring
        scaler = MinMaxScaler()
        metrics_to_normalize = ['CompletionRate', 'Duration', 'PeakHourRatio', 'CancellationRate']
        
        # Ensure all required columns exist
        for metric in metrics_to_normalize:
            if metric not in room_metrics.columns:
                print(f"Warning: {metric} not found in metrics. Available columns: {room_metrics.columns.tolist()}")
                return None
        
        normalized_metrics = pd.DataFrame(
            scaler.fit_transform(room_metrics[metrics_to_normalize]),
            columns=metrics_to_normalize
        )
        
        # Calculate room demand score (higher score = more demand)
        room_metrics['DemandScore'] = (
            normalized_metrics['CompletionRate'] * 0.4 +
            normalized_metrics['Duration'] * 0.3 +
            normalized_metrics['PeakHourRatio'] * 0.3
        )
        
        # Calculate cancellation penalty factor (higher for rooms with more cancellations)
        room_metrics['CancellationPenalty'] = (
            normalized_metrics['CancellationRate'] * 0.7 +
            (1 - normalized_metrics['CompletionRate']) * 0.3
        )
        
        # Calculate recommended fee as percentage of hourly rate
        # Base percentage: 10-30% depending on demand
        # Additional penalty: 0-20% based on cancellation history
        room_metrics['BaseFeePercentage'] = (1 - room_metrics['DemandScore']) * 20 + 10
        room_metrics['PenaltyPercentage'] = room_metrics['CancellationPenalty'] * 20
        room_metrics['RecommendedFeePercentage'] = (
            room_metrics['BaseFeePercentage'] + room_metrics['PenaltyPercentage']
        ).clip(10, 50)  # Cap between 10% and 50%
        
        # Calculate recommended fee (based on hourly rate, not total cost)
        room_metrics['RecommendedFee'] = (
            room_metrics['PricePerHour'] * room_metrics['RecommendedFeePercentage'] / 100
        )
        
        # Ensure fee doesn't exceed 50% of average booking cost
        room_metrics['RecommendedFee'] = np.minimum(
            room_metrics['RecommendedFee'],
            room_metrics['TotalCost'] * 0.5
        )
        
        return room_metrics

    def get_workspace_recommendations(self, workspace_id):
        """Generate cancellation fee recommendations for a specific workspace"""
        try:
            self.connect_to_db()
            
            # Get and analyze data for specific workspace
            booking_data = self.get_room_booking_data(workspace_id)
            room_metrics = self.analyze_booking_patterns(booking_data)
            
            if room_metrics is None:
                print(f"No data found for workspace ID: {workspace_id}")
                return None
                
            recommendations = self.calculate_cancellation_fees(room_metrics)
            
            if recommendations is None:
                print("Error: Failed to calculate cancellation fees")
                return None
            
            # Sort by recommended fee
            recommendations = recommendations.sort_values('RecommendedFee', ascending=False)
            
            # Format output
            output = recommendations[[
                'room_name', 'room_type', 'PricePerHour', 'CompletionRate',
                'CancellationRate', 'RecommendedFeePercentage', 'RecommendedFee'
            ]].round(2)
            
            return output
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return None
        finally:
            self.close_connection()

    def generate_recommendations(self):
        """Generate cancellation fee recommendations for all workspaces"""
        try:
            self.connect_to_db()
            
            # Get and analyze data
            booking_data = self.get_room_booking_data()
            room_metrics = self.analyze_booking_patterns(booking_data)
            
            if room_metrics is None:
                print("Error: Failed to analyze booking patterns")
                return None
                
            recommendations = self.calculate_cancellation_fees(room_metrics)
            
            if recommendations is None:
                print("Error: Failed to calculate cancellation fees")
                return None
            
            # Sort by recommended fee
            recommendations = recommendations.sort_values('RecommendedFee', ascending=False)
            
            # Format output
            output = recommendations[[
                'workspace_name', 'room_name', 'room_type', 'PricePerHour', 
                'CompletionRate', 'CancellationRate', 'RecommendedFeePercentage', 
                'RecommendedFee'
            ]].round(2)
            
            return output
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return None
        finally:
            self.close_connection()
