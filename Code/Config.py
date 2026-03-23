import os


class Config:
    """Central configuration for the CA project."""

    TICKET_SUMMARY = "Ticket Summary"
    INTERACTION_CONTENT = "Interaction content"
    TICKET_ID = "Ticket id"

    Y1 = "y1"
    Y2 = "y2"
    Y3 = "y3"
    Y4 = "y4"

    CLASS_COL = Y2
    GROUPED = Y1

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    DATA_FILES = ["AppGallery.csv", "Purchasing.csv"]

    COLUMN_RENAME = {
        "Type 1": Y1,
        "Type 2": Y2,
        "Type 3": Y3,
        "Type 4": Y4,
    }

    CHAIN_SEPARATOR = " | "
    CHAINED_TARGETS = {
        "chain_1": [Y2],
        "chain_2": [Y2, Y3],
        "chain_3": [Y2, Y3, Y4],
    }

    HIERARCHICAL_LEVELS = [Y2, Y3, Y4]

    MIN_CLASS_COUNT = 5
    MIN_SUBSET_SIZE = 10
    MIN_BRANCH_CLASS_COUNT = 2

    N_ESTIMATORS = 1000
    TEST_SIZE = 0.20
    SEED = 0

    MAX_FEATURES = 2000
    MIN_DF = 4
    MAX_DF = 0.90

    ENABLE_TRANSLATION = False
