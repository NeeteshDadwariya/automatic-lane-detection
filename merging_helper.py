from math import atan2, degrees, sqrt, atan


def get_orientation(line):
    '''get orientation of a line, using its length
    https://en.wikipedia.org/wiki/Atan2
    '''
    orientation = atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
    return degrees(orientation)


def checker(line_new, groups, min_distance_to_merge, max_distance_to_avoid_merge, min_angle_to_merge):
    '''Check if line have enough distance and angle to be count as similar
    '''

    for group in groups:
        # walk through existing line groups
        for line_old in group:
            min_dist, max_dist = get_distance(line_old, line_new)
            # if max_dist >= max_distance_to_avoid_merge:
            #     return True

            if min_dist <= min_distance_to_merge:
                # check the angle between lines
                orientation_new = get_orientation(line_new)
                orientation_old = get_orientation(line_old)
                # if all is ok -- line is similar to others in group
                if abs(orientation_new - orientation_old) < min_angle_to_merge:
                    group.append(line_new)
                    return False

    # if it is totally different line
    return True


def DistancePointLine(point, line):
    """Get distance between point and line
    http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    """
    px, py = point
    x1, y1, x2, y2 = line

    def lineMagnitude(x1, y1, x2, y2):
        'Get line (aka vector) length'
        lineMagnitude = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
        return lineMagnitude

    LineMag = lineMagnitude(x1, y1, x2, y2)
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        # // closest point does not fall within the line segment, take the shorter distance
        # // to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine

    # p1 = np.array([x1, y1])
    # p2 = np.array([x2, y2])
    # p3 = np.array([px, py])
    # #return np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    # return abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1)) / np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))


def get_distance(a_line, b_line):
    """Get all possible distances between each dot of two lines and second line
    return the shortest
    """
    dist1 = DistancePointLine(a_line[:2], b_line)
    dist2 = DistancePointLine(a_line[2:], b_line)
    dist3 = DistancePointLine(b_line[:2], a_line)
    dist4 = DistancePointLine(b_line[2:], a_line)

    return min(dist1, dist2, dist3, dist4), max(dist1, dist2, dist3, dist4)


def merge_lines_pipeline_2(lines):
    'Clusterize (group) lines'
    groups = []  # all lines groups are here
    # Parameters to play with
    min_distance_to_merge = 10
    max_distance_to_avoid_merge = 400
    min_angle_to_merge = 5
    # first line will create new group every time
    groups.append([lines[0]])
    # if line is different from existing gropus, create a new group
    for line_new in lines[1:]:
        if checker(line_new, groups, min_distance_to_merge, max_distance_to_avoid_merge, min_angle_to_merge):
            groups.append([line_new])

    return groups


def merge_lines_segments1(lines):
    """Sort lines cluster and return first and last coordinates
    """
    orientation = get_orientation(lines[0])

    # special case
    if (len(lines) == 1):
        return [lines[0][:2], lines[0][2:]]

    # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
    points = []
    for line in lines:
        points.append(line[:2])
        points.append(line[2:])
    # if vertical
    if 45 < orientation < 135:
        # sort by y
        points = sorted(points, key=lambda point: point[1])
    else:
        # sort by x
        points = sorted(points, key=lambda point: point[0])

    # return first and last point in sorted group
    # [[x,y],[x,y]]
    return [points[0], points[-1]]


def merge_hough_lines(lines):
    # cut_lines = []
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     line_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    #     x11 = x1
    #     y11 = y1
    #     if line_length>350:
    #         x1 = int((2*x1)/3 + (x2/3))
    #         y1 = int((2*y1)/3 + (y2/3))
    #
    #         x2 = int((2*x2)/3 + (x11/3))
    #         y2 = int((2*y2)/3 + (y11/3))
    #
    #     cut_lines.append([[x1, y1, x2, y2]])

    lines = lines

    lines_x = []
    lines_y = []

    # for every line of cv2.HoughLinesP()
    for line_i in [l[0] for l in lines]:
        orientation = get_orientation(line_i)
        # if vertical
        if 45 < orientation < 135:
            lines_y.append(line_i)
        else:
            lines_x.append(line_i)

    lines_y = sorted(lines_y, key=lambda line: line[1])
    lines_x = sorted(lines_x, key=lambda line: line[0])
    merged_lines_all = []

    # for each cluster in vertical and horizantal lines leave only one line
    for i in [lines_x, lines_y]:
        if len(i) > 0:
            groups = merge_lines_pipeline_2(i)
            merged_lines = []
            for group in groups:
                merged_lines.append(merge_lines_segments1(group))

            merged_lines_all.extend(merged_lines)

    final_lines = []
    for line in merged_lines_all:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[1][0]
        y2 = line[1][1]
        final_lines.append([[x1, y1, x2, y2]])

    return final_lines


def get_slope(x1, y1, x2, y2):
    if x2 == x1:
        return float('inf')
    return (y2 - y1) / (x2 - x1)


def get_intercept(slope, x1, y1):
    return y1 - (slope * x1)


def filter_lines(hough_lines):
    slopes = []
    intercepts = []
    merged_lines = []
    for line in hough_lines:
        x1, y1, x2, y2 = line
        slope = get_slope(x1, y1, x2, y2)
        intercept = get_intercept(slope, x1, y1)
        line_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = abs(degrees(atan(slope)))
        if all((12 <= angle <= 89, line_length > 150)):
            merged_lines.append((x1, y1, x2, y2))
            slopes.append(slope)
            intercepts.append(intercept)
    return merged_lines
