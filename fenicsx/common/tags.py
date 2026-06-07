from enum import IntEnum

class Tags(IntEnum):

    # ==========================================
    # Subdomains
    # ==========================================

    FLUID = 10
    SOLID = 20

    # ==========================================
    # Boundaries
    # ==========================================

    CYLINDER = 101

    INTERFACE = 102

    CLAMP = 103

    INLET = 104
    OUTLET = 105

    BOTTOM_WALL = 106
    TOP_WALL = 107
