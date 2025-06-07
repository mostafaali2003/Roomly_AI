# fallback_recommendations.py
import random

# List of 10 different room recommendations for fallback
FALLBACK_RECOMMENDATIONS = [
    {
        "type": "Executive Boardroom",
        "description": "Premium meeting space with high-end furnishings and technology for important client meetings and leadership discussions.",
        "capacity": "12-16",
        "price_range": "$40-60/hr",
        "benefits": "Professional ambiance, integrated AV system, catering options"
    },
    {
        "type": "Focus Pod",
        "description": "Individual soundproof booth for concentrated work, virtual meetings, or private phone calls without distractions.",
        "capacity": "1",
        "price_range": "$10-15/hr",
        "benefits": "Privacy, noise cancellation, ergonomic setup"
    },
    {
        "type": "Creative Studio",
        "description": "Versatile space with movable furniture, whiteboards, and brainstorming tools designed for ideation and creative collaboration.",
        "capacity": "6-10",
        "price_range": "$30-45/hr",
        "benefits": "Flexible layout, creative atmosphere, digital and analog tools"
    },
    {
        "type": "Presentation Theater",
        "description": "Tiered seating arrangement with professional audio/visual equipment for presentations, training sessions, and small events.",
        "capacity": "20-30",
        "price_range": "$50-80/hr",
        "benefits": "Professional presentation setup, audience engagement, recording capabilities"
    },
    {
        "type": "Wellness Room",
        "description": "Calming space designed for meditation, rest, or wellness activities that promote employee well-being and stress reduction.",
        "capacity": "4-6",
        "price_range": "$20-30/hr",
        "benefits": "Relaxation, work-life balance, employee wellness"
    },
    {
        "type": "Innovation Lab",
        "description": "Tech-equipped workspace with prototyping tools and resources for developing and testing new ideas and products.",
        "capacity": "8-12",
        "price_range": "$35-55/hr",
        "benefits": "Innovation support, collaborative environment, specialized equipment"
    },
    {
        "type": "Networking Lounge",
        "description": "Casual meeting area with comfortable seating and refreshment options for informal discussions and professional networking.",
        "capacity": "10-15",
        "price_range": "$25-40/hr",
        "benefits": "Casual atmosphere, relationship building, comfortable seating"
    },
    {
        "type": "Training Classroom",
        "description": "Structured learning environment with tables, chairs, and educational technology for workshops, classes, and professional development.",
        "capacity": "15-25",
        "price_range": "$35-50/hr",
        "benefits": "Learning-focused setup, educational technology, group activities"
    },
    {
        "type": "Podcast Studio",
        "description": "Acoustically treated room with recording equipment for creating professional audio content, podcasts, and voice recordings.",
        "capacity": "2-4",
        "price_range": "$30-45/hr",
        "benefits": "Professional audio production, content creation, sound isolation"
    },
    {
        "type": "Virtual Reality Room",
        "description": "Dedicated space with VR equipment and ample movement area for immersive presentations, training, or entertainment experiences.",
        "capacity": "4-8",
        "price_range": "$40-65/hr",
        "benefits": "Cutting-edge technology, immersive experiences, unique client offerings"
    }
]

def get_random_recommendations(count=3):
    """Return a specified number of random recommendations from the fallback list"""
    if count >= len(FALLBACK_RECOMMENDATIONS):
        return FALLBACK_RECOMMENDATIONS
    
    # Randomly select 'count' recommendations without replacement
    selected_recommendations = random.sample(FALLBACK_RECOMMENDATIONS, count)
    return selected_recommendations