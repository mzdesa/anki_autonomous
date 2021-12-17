from sampling_MPC import pointInRotatedRectangle, convert_colinear2xy
import numpy as np
from obstacle import Obstacle
from planning import intersect

def test_pointInRotatedRectangle():
    pointX = -1
    pointY = 2
    rectX = 0
    rectY = 0
    rectWidth = 4
    rectHeight = 2
    rectAngle = np.pi/4
    assert(pointInRotatedRectangle(pointX, pointY, rectX, rectY, rectWidth, rectHeight, rectAngle) == False)

    pointX = -1
    pointY = 2 
    rectX = 0
    rectY = 0 
    rectWidth = 4
    rectHeight = 2 
    rectAngle = 0
    assert(pointInRotatedRectangle(pointX, pointY, rectX, rectY, rectWidth, rectHeight, rectAngle) == True)

    pointX = -2
    pointY = -2
    rectX = 1
    rectY = 0
    rectWidth = 2
    rectHeight = 4
    rectAngle = 0
    assert(pointInRotatedRectangle(pointX, pointY, rectX, rectY, rectWidth, rectHeight, rectAngle) == False)

def test_convert_colinear2xy(): #s.n to x,y
    # print(convert_colinear2xy(216, 2)) #piece 0 --> (2, -10.1)
    # print(convert_colinear2xy(7, 18)) #piece 1: (7, 18) --> (10,18) 
    # print(convert_colinear2xy(40, 11)) #piece 2: (-37, 20)   
    # print(convert_colinear2xy(66.55, 5)) #piece 3: (half circle + 10, 5) --> (-41, -10) 
    print(convert_colinear2xy(np.pi*18 + 56.5 + 13, 11)) # piece 4: (-39, -56.6 - 19)     

def test_collision():
    assert(intersect([6, -40], [6, -31.5], [0, -34], [8, -34]) == True)
    assert(intersect([8, -34], [8, -26], [6,-31.5], [10.5, -31.5]) == True)

if __name__ == "__main__":
    test_pointInRotatedRectangle()
    # test_convert_colinear2xy()
    test_collision()