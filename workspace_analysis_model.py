from mysql.connector import Error
import pandas as pd
from typing import Dict, List, Optional, Union
from tabulate import tabulate
import mysql.connector
from mysql.connector import pooling
import os

class WorkspaceDashboard:
    def __init__(self, host: str, database: str, user: str, password: str, connection_pool=None):
        """Initialize database connection parameters"""
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection_pool = connection_pool
        self.connection = None

    def connect(self) -> bool:
        """Establish connection to MySQL database"""
        try:
            if self.connection_pool:
                self.connection = self.connection_pool.get_connection()
            else:
                self.connection = mysql.connector.connect(
                    host=self.host,
                    database=self.database,
                    user=self.user,
                    password=self.password
                )
            
            if self.connection.is_connected():
                print("Successfully connected to MySQL database")
                return True
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            return False
        return False

    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection closed")

    def execute_query(self, query: str, params: tuple = None) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results as pandas DataFrame"""
        try:
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    return None

            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            results = cursor.fetchall()
            cursor.close()
            
            if results:
                return pd.DataFrame(results)
            return None
        except Error as e:
            print(f"Error executing query: {e}")
            return None

    def get_workspace_info(self, workspace_id: str) -> Optional[pd.DataFrame]:
        """Get basic workspace information"""
        query = """
        SELECT 
            w.*,
            l.City,
            l.Town,
            l.Country,
            wp.YearPrice,
            wp.MonthPrice
        FROM Workspace w
        LEFT JOIN Location l ON w.LocationId = l.Id
        LEFT JOIN WorkspacePlan wp ON w.Id = wp.WorkspaceId
        WHERE w.Id = %s
        """
        return self.execute_query(query, (workspace_id,))

    def get_room_statistics(self, workspace_id: str) -> Optional[pd.DataFrame]:
        """Get room statistics"""
        query = """
        SELECT 
            COUNT(*) as TotalRooms,
            SUM(Capacity) as TotalCapacity,
            SUM(AvailableCount) as TotalAvailableRooms,
            AVG(PricePerHour) as AveragePricePerHour,
            COUNT(CASE WHEN RoomStatus = 'Available' THEN 1 END) as AvailableRooms,
            COUNT(CASE WHEN RoomStatus = 'Occupied' THEN 1 END) as OccupiedRooms
        FROM Room
        WHERE WorkspaceId = %s
        """
        return self.execute_query(query, (workspace_id,))

    def get_room_types_distribution(self, workspace_id: str) -> Optional[pd.DataFrame]:
        """Get room types distribution"""
        query = """
        SELECT 
            Type,
            COUNT(*) as RoomCount,
            AVG(PricePerHour) as AveragePrice,
            SUM(Capacity) as TotalCapacity
        FROM Room
        WHERE WorkspaceId = %s
        GROUP BY Type
        """
        return self.execute_query(query, (workspace_id,))

    def get_recent_reviews(self, workspace_id: str, limit: int = 10) -> Optional[pd.DataFrame]:
        """Get recent reviews and ratings"""
        query = """
        SELECT 
            r.*,
            u.FName,
            u.LName
        FROM Review r
        JOIN User u ON r.UserId = u.Id
        WHERE r.WorkspaceId = %s
        ORDER BY r.ReviewDate DESC
        LIMIT %s
        """
        return self.execute_query(query, (workspace_id, limit))

    def get_active_bookings(self, workspace_id: str) -> Optional[pd.DataFrame]:
        """Get current active bookings"""
        query = """
        SELECT 
            b.*,
            r.StartTime,
            r.EndTime,
            r.Status as ReservationStatus,
            rm.Name as RoomName,
            u.FName,
            u.LName
        FROM Booking b
        JOIN Reservation r ON b.ReservationId = r.Id
        JOIN Room rm ON b.RoomId = rm.Id
        JOIN User u ON b.UserId = u.Id
        WHERE b.WorkspaceId = %s
        AND r.Status = 'Active'
        AND r.StartTime <= NOW()
        AND r.EndTime >= NOW()
        """
        return self.execute_query(query, (workspace_id,))

    def get_revenue_statistics(self, workspace_id: str) -> Optional[pd.DataFrame]:
        """Get revenue statistics"""
        query = """
        SELECT 
            SUM(p.Amount) as TotalRevenue,
            COUNT(DISTINCT p.Id) as TotalPayments,
            AVG(p.Amount) as AveragePaymentAmount
        FROM Payment p
        JOIN Reservation r ON p.ReservationId = r.Id
        JOIN Booking b ON r.Id = b.ReservationId
        WHERE b.WorkspaceId = %s
        AND p.Status = 'Completed'
        """
        return self.execute_query(query, (workspace_id,))

    def get_monthly_revenue_trend(self, workspace_id: str, months: int = 12) -> Optional[pd.DataFrame]:
        """Get monthly revenue trend"""
        query = """
        SELECT 
            DATE_FORMAT(p.PaymentDate, '%Y-%m') as Month,
            SUM(p.Amount) as MonthlyRevenue
        FROM Payment p
        JOIN Reservation r ON p.ReservationId = r.Id
        JOIN Booking b ON r.Id = b.ReservationId
        WHERE b.WorkspaceId = %s
        AND p.Status = 'Completed'
        GROUP BY DATE_FORMAT(p.PaymentDate, '%Y-%m')
        ORDER BY Month DESC
        LIMIT %s
        """
        return self.execute_query(query, (workspace_id, months))

    def get_active_offers(self, workspace_id: str) -> Optional[pd.DataFrame]:
        """Get active offers"""
        query = """
        SELECT 
            o.*,
            ao.RoomId
        FROM Offers o
        JOIN ApplyedOffers ao ON o.Id = ao.OfferId
        JOIN Room r ON ao.RoomId = r.Id
        WHERE r.WorkspaceId = %s
        AND o.Status = 'Active'
        AND o.ValidFrom <= CURDATE()
        AND o.ValidTo >= CURDATE()
        """
        return self.execute_query(query, (workspace_id,))

    def get_amenities_overview(self, workspace_id: str) -> Optional[pd.DataFrame]:
        """Get amenities overview"""
        query = """
        SELECT 
            a.Type,
            COUNT(*) as AmenityCount,
            SUM(a.TotalCount) as TotalItems
        FROM Amenity a
        JOIN Room r ON a.RoomId = r.Id
        WHERE r.WorkspaceId = %s
        GROUP BY a.Type
        """
        return self.execute_query(query, (workspace_id,))

    def get_workspace_analytics(self, workspace_id: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get workspace analytics"""
        query = """
        SELECT 
            wa.*,
            w.Name as WorkspaceName
        FROM WorkspaceAnalytics wa
        JOIN Workspace w ON wa.WorkspaceId = w.Id
        WHERE wa.WorkspaceId = %s
        ORDER BY wa.Date DESC
        LIMIT %s
        """
        return self.execute_query(query, (workspace_id, days))

    def get_popular_rooms(self, workspace_id: str, limit: int = 5) -> Optional[pd.DataFrame]:
        """Get popular rooms"""
        query = """
        SELECT 
            r.*,
            COUNT(b.RoomId) as BookingCount
        FROM Room r
        LEFT JOIN Booking b ON r.Id = b.RoomId
        WHERE r.WorkspaceId = %s
        GROUP BY r.Id
        ORDER BY BookingCount DESC
        LIMIT %s
        """
        return self.execute_query(query, (workspace_id, limit))

    def get_staff_information(self, workspace_id: str) -> Optional[pd.DataFrame]:
        """Get staff information"""
        query = """
        SELECT 
            ws.*,
            w.Name as WorkspaceName
        FROM WorkspaceStaff ws
        JOIN WorkspaceSupervise wsup ON ws.Id = wsup.StaffId
        JOIN Workspace w ON wsup.WorkspaceId = w.Id
        WHERE w.Id = %s
        """
        return self.execute_query(query, (workspace_id,))

    def get_all_statistics(self, workspace_id: str) -> dict:
        """Get all statistics for a workspace"""
        # Connect to database
        if not self.connect():
            return {"error": "Failed to connect to database"}

        try:
            # Collect all statistics
            statistics = {
                "workspace_info": self.get_workspace_info(workspace_id),
                "room_statistics": self.get_room_statistics(workspace_id),
                "room_types_distribution": self.get_room_types_distribution(workspace_id),
                "recent_reviews": self.get_recent_reviews(workspace_id),
                "active_bookings": self.get_active_bookings(workspace_id),
                "revenue_statistics": self.get_revenue_statistics(workspace_id),
                "monthly_revenue_trend": self.get_monthly_revenue_trend(workspace_id),
                "active_offers": self.get_active_offers(workspace_id),
                "amenities_overview": self.get_amenities_overview(workspace_id),
                "workspace_analytics": self.get_workspace_analytics(workspace_id),
                "popular_rooms": self.get_popular_rooms(workspace_id),
                "staff_information": self.get_staff_information(workspace_id)
            }

            # Convert DataFrames to dictionaries
            result = {}
            for key, value in statistics.items():
                if value is not None and not value.empty:
                    result[key] = value.to_dict(orient='records')
                else:
                    result[key] = []

            return result

        except Exception as e:
            return {"error": str(e)}

        finally:
            # Always disconnect when done
            self.disconnect()

    def print_all_statistics(self, workspace_id: str):
        """Print all available statistics for the workspace in a formatted way"""
        print("\n" + "="*80)
        print(f"WORKSPACE DASHBOARD STATISTICS - Workspace ID: {workspace_id}")
        print("="*80)

        # 1. Basic Workspace Information
        print("\n1. BASIC WORKSPACE INFORMATION")
        print("-"*40)
        workspace_info = self.get_workspace_info(workspace_id)
        if workspace_info is not None and not workspace_info.empty:
            print(tabulate(workspace_info, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No workspace information found")

        # 2. Room Statistics
        print("\n2. ROOM STATISTICS")
        print("-"*40)
        room_stats = self.get_room_statistics(workspace_id)
        if room_stats is not None and not room_stats.empty:
            print(tabulate(room_stats, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No room statistics found")

        # 3. Room Types Distribution
        print("\n3. ROOM TYPES DISTRIBUTION")
        print("-"*40)
        room_types = self.get_room_types_distribution(workspace_id)
        if room_types is not None and not room_types.empty:
            print(tabulate(room_types, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No room types distribution found")

        # 4. Recent Reviews
        print("\n4. RECENT REVIEWS")
        print("-"*40)
        reviews = self.get_recent_reviews(workspace_id)
        if reviews is not None and not reviews.empty:
            print(tabulate(reviews, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No recent reviews found")

        # 5. Active Bookings
        print("\n5. ACTIVE BOOKINGS")
        print("-"*40)
        active_bookings = self.get_active_bookings(workspace_id)
        if active_bookings is not None and not active_bookings.empty:
            print(tabulate(active_bookings, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No active bookings found")

        # 6. Revenue Statistics
        print("\n6. REVENUE STATISTICS")
        print("-"*40)
        revenue_stats = self.get_revenue_statistics(workspace_id)
        if revenue_stats is not None and not revenue_stats.empty:
            print(tabulate(revenue_stats, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No revenue statistics found")

        # 7. Monthly Revenue Trend
        print("\n7. MONTHLY REVENUE TREND")
        print("-"*40)
        revenue_trend = self.get_monthly_revenue_trend(workspace_id)
        if revenue_trend is not None and not revenue_trend.empty:
            print(tabulate(revenue_trend, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No monthly revenue trend found")

        # 8. Active Offers
        print("\n8. ACTIVE OFFERS")
        print("-"*40)
        active_offers = self.get_active_offers(workspace_id)
        if active_offers is not None and not active_offers.empty:
            print(tabulate(active_offers, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No active offers found")

        # 9. Amenities Overview
        print("\n9. AMENITIES OVERVIEW")
        print("-"*40)
        amenities = self.get_amenities_overview(workspace_id)
        if amenities is not None and not amenities.empty:
            print(tabulate(amenities, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No amenities overview found")

        # 10. Workspace Analytics
        print("\n10. WORKSPACE ANALYTICS")
        print("-"*40)
        analytics = self.get_workspace_analytics(workspace_id)
        if analytics is not None and not analytics.empty:
            print(tabulate(analytics, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No workspace analytics found")

        # 11. Popular Rooms
        print("\n11. POPULAR ROOMS")
        print("-"*40)
        popular_rooms = self.get_popular_rooms(workspace_id)
        if popular_rooms is not None and not popular_rooms.empty:
            print(tabulate(popular_rooms, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No popular rooms found")

        # 12. Staff Information
        print("\n12. STAFF INFORMATION")
        print("-"*40)
        staff_info = self.get_staff_information(workspace_id)
        if staff_info is not None and not staff_info.empty:
            print(tabulate(staff_info, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No staff information found")
