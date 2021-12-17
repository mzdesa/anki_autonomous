#oval track with two straight sections and two turns. Parametrize as a space curve WRT time.
#solve for offset and arc length coords
import numpy as np
from scipy import optimize

def coord_solver(ar_state1, car_state, r, straight_len):
    """
    Solve for absolute s, n coordinates of car object
    car - overdrive object
    ar_state1- (x, y, z) of ar tag at the start of the 1st turn
    car_state - (x, y, z) of car
    r - track turn radius
    straight_len - length of straight section
    """
    x = car_state[0]
    y = car_state[1]
    piece = 0
    if piece == 0: #car.piece == 0:
        print('straight')
        s = (y - ar_state1[1])+straight_len + (straight_len + 2*np.pi*r) #have offset values to avoid negative
        n = x-ar_state1[0]
    elif piece == 1 or piece == 2:                       
        print("turn")
        n = np.sqrt((x + r)**2 + y**2) - r
        arr1 = np.array([x+r, y])
        arr2 = np.array([r, 0])
        theta = np.arccos(np.dot(arr1, arr2) / (r*np.linalg.norm(arr1)))
        s = r * theta
    elif piece == 3:
        print('straight')
        s = abs(y) + np.pi*r
        n = -(x + 2*r)
    elif piece == 4 or piece == 5:
        print('turn')
        n = np.sqrt((x + r)**2 + (y + straight_len)**2) - r
        arr1 = np.array([x+r, y+straight_len])
        arr2 = np.array([-r, 0])
        theta = np.arccos(np.dot(arr1, arr2) / (np.linalg.norm(arr2)*np.linalg.norm(arr1)))
        s = r * theta + straight_len + np.pi*r
    return s, n


if __name__ == "__main__":
    arr_state1 = [0, 0, 0]
    car_state = [7, -17, 0]
    r = 13.25/2
    straight_len = 22
    print(coord_solver(arr_state1, car_state, r, straight_len))