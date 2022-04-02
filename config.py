from dotmap import DotMap

expected_line_stats = {
    'lane1.jpeg': 3,
    'lane2.jpeg': 6,
    'lane3.jpeg': 8,
    'lane4.jpeg': 3,
    'lane5_stop.jpeg': 3,
    'lane6.jpeg': 8,
    'lane7.jpeg': 8,
    'fire-lane.jpeg': 2,
    'fire-lane2.jpeg': 2,
    'stop.jpeg': 0,
}

config = DotMap()
config.EROSION_KERNEL_SIZE = 2
config.DILATION_KERNEL_SIZE = 2
config.ITERATIONS = 1
config.MIN_CONTOUR_AREA = 100
config.GAUSSIAN_KERNEL_SIZE = 5
config.OPENING_KERNEL_SIZE = 2
config.HOUGH_THRESHOLD = 60
config.HOUGH_MIN_LINE_LENGTH = 130
config.HOUGH_MAX_LINE_GAP = 35
config.MAX_REGION_HEIGHT = .7
config.LINE_MERGE_COUNT = 2
