#!/usr/bin/env python

import time
from ankicar import Ankicar
import numpy as np
import random
import datetime
import casadi as ca
from obstacle import Obstacle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
from mapping_track_to_model import *
from matplotlib.transforms import Affine2D


def get_track_piece(s):
    # print(s)
    r = 18.0  # c
    straight_len = 56.5  # cm
    circum = 2.0 * np.pi * r
    s = s % 226
    if s >= 0 and s < circum/4:
        return 1
    elif s >= circum/4 and s < circum/2:
        return 2
    elif s >= circum/2 and s < circum/2 + straight_len:
        return 3
    elif s >= circum/2 + straight_len and s < 3 * circum/4 + straight_len:
        return 4
    elif s >= 3 * circum/4 + straight_len and s < circum + straight_len:
        return 5
    elif s >= circum + straight_len and s < circum + 2 * straight_len:
        return 0
    return


def convert_curvilinear2xy(s, offset):
    r = 18.0  # cm
    straight_len = 56.5  # cm
    circum = 2 * np.pi * r
    track_piece = get_track_piece(s)
    # print("track piece: " + str(track_piece))
    if track_piece == 0:
        return offset, -(straight_len - (s - (circum + straight_len)))
    if track_piece == 1 or track_piece == 2:  # might need to separate
        theta = abs(s/r)
        return (r + offset)*np.cos(theta) - r, (r + offset) * np.sin(theta)
    if track_piece == 3:
        return - 2 * r - offset, - (s - circum/2)
    if track_piece == 4 or track_piece == 5:
        theta = abs((s - (circum/2 + straight_len))/r)
        # print(theta * 180/np.pi)
        return - r - (r + offset) * np.cos(theta), -straight_len - (r+offset) * np.sin(theta)
    return


def visualize(points, obstacle, car, debug=None):
    x_arr, y_arr = [], []
    fig, ax = plt.subplots()
    plt.scatter([0], [0], color='r')
    # print(points)
    for i in range(points.shape[1]):
        x, y = convert_curvilinear2xy(points[0][i], points[1][i])
        x_arr.append(x)
        y_arr.append(y)

    ax.add_patch(Rectangle(
        (car.curr_xy[0] - (car.width/2), car.curr_xy[1] - (car.length/2)), car.width, car.length))
    ax.add_patch(Rectangle((obstacle.x - (obstacle.width/2), obstacle.y -
                 (obstacle.length/2)), obstacle.width, obstacle.length))

    if debug != None:
        for rect in debug:
            print("DRAWING RECTS")
            point = rect[0]
            ax.add_patch(
                Rectangle((point[0] - (car.width/2), point[1] - (car.length/2)), 4.5, 8.5))

    ir, otr, straight = 18, 22.5 + 18, 56.5
    # --Draw inner top
    x = np.linspace(-ir + -ir, ir + -ir)
    y = np.sqrt(ir**2 - ((x + ir)) ** 2)
    ax.plot(x, y, c='orange')
    # --
    # --Draw outer top
    x = np.linspace((-otr - ir), (otr - ir))
    y = np.sqrt((otr)**2 - ((x + ir)) ** 2)
    ax.plot(x, y, c='orange')
    # --

    # --Draw inner bottom
    x = np.linspace(-ir + -ir, ir + -ir)
    y = -np.sqrt(ir**2 - ((x + ir)) ** 2) - straight
    ax.plot(x, y, c='orange')
    # --
    # --Draw outer bottom
    x = np.linspace((-otr - ir), (otr - ir))
    y = -np.sqrt((otr)**2 - ((x + ir)) ** 2) - straight
    ax.plot(x, y, c='orange')
    # --
    y = np.linspace(-straight, 0)
    x = y * 0 + 0
    ax.plot(x, y, c='orange')
    y = np.linspace(-straight, 0)
    x = y * 0 + (otr - ir)
    ax.plot(x, y, c='orange')
    y = np.linspace(-straight, 0)
    x = y * 0 + (- 2 * ir)
    ax.plot(x, y, c='orange')
    y = np.linspace(-straight, 0)
    x = y * 0 + (- ir - otr)
    ax.plot(x, y, c='orange')

    plt.scatter(x_arr[0], y_arr[0], c='g', zorder=2)
    plt.scatter(x_arr[1:], y_arr[1:], c='k', zorder=2)
    plt.axis('square')
    plt.savefig("optimal_traj_%f.png" % time.time())

    return


def best_traj_pts(r, theta, horizon, car, timestep, obstacle, x_dim=5):
    # get best points for next 5 points not including the current point
    def s_dot_solver(s_curr, s_prev):
        if s_curr-s_prev < -100:  # define -100 as a threshold value
            return (s_curr+226.097-s_prev)/timestep
        return (s_curr-s_prev)/timestep

    def n_dot_solver(n_curr, n_prev): return (n_curr-n_prev)/timestep

    reference_pts = np.zeros((x_dim, horizon+1))
    state = car.curr_state
    s0 = state[0]
    n0 = state[1]
    s_dot = state[2]
    n_dot = state[3]
    car_theta = state[4]
    s1 = s0 + abs(r)*np.sin(theta)
    n1 = n0 - r * (1 - np.cos(theta))
    s2 = s1 + abs(r)*np.sin(theta)
    n2 = n1 - r * (1 - np.cos(theta))
    d01 = r*theta
    d02 = 2*d01
    reference_pts[:, 0] = np.array(
        [s0 % 226.07, n0, s_dot, n_dot, car_theta]).squeeze()
    v = np.sqrt(s_dot**2 + n_dot**2)
    d = 0
    s_prev = s0
    n_prev = n0

    for i in range(1, horizon+1):
        d = d + v*timestep
        s = None
        n = None
        t = None
        if d < d01:
            # project onto curve 1
            t = d/r
            s = s0 + abs(r*np.sin(t))
            n = n0 - r * (1 - np.cos(t))

        elif d < d02:
            d_remaining = d - d01
            t = d_remaining/r
            s = s1 + abs(r*np.sin(t))
            n = n1 - r * (1 - np.cos(t))
        else:
            d_remaining = d - d02
            t = 0
            s = s2 + d_remaining
            n = n2
        # redefine v down here
        s_dot = s_dot_solver(s, s_prev)
        n_dot = n_dot_solver(n, n_prev)
        s_prev = s
        n_prev = n
        phi = np.arctan2(n_dot, s_dot)
        v = np.sqrt(s_dot**2 + n_dot**2)
        reference_pts[:, i] = np.array(
            [s % 226.07, n, s_dot, n_dot, phi]).squeeze()

    return reference_pts


def if_collision(rad, theta, obstacle, car):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    # Return true if line segments AB and CD intersect
    def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    com_car_tested = np.zeros((15, 2))

    state = car.curr_state 
    s0 = state[0]
    n0 = state[1]
    s1 = s0 + abs(rad)*np.sin(theta)
    n1 = n0 - rad * (1 - np.cos(theta))
    s2 = s1 + abs(rad)*np.sin(theta)
    n2 = n1 - rad * (1 - np.cos(theta))
    car_width, car_height = 4.5, 8.5  # cm | x dxn, y dxn #use car lengths
    obs_x = obstacle.x
    obs_y = obstacle.y
    obs_width = obstacle.width + obstacle.buffer
    obs_length = obstacle.length + obstacle.buffer
    obs_corners = [[obs_x - obs_width/2, obs_y + obs_length/2],
                   [obs_x + obs_width/2, obs_y + obs_length/2],
                   [obs_x + obs_width/2, obs_y - obs_length/2],
                   [obs_x - obs_width/2, obs_y - obs_length/2]]

    
    test_thetas = [i*(theta/4) for i in range(1,5)]
    count = 0
    for t in test_thetas: #curve 1 section
        car_s, car_n = s0 + abs(rad*np.sin(t)), n0 - rad * (1 - np.cos(t))
        car_x, car_y = convert_curvilinear2xy(car_s, car_n)
        car_corners = [[car_x - car_width/2, car_y + car_height/2],
                       [car_x + car_width/2, car_y + car_height/2],
                       [car_x + car_width/2, car_y - car_height/2],
                       [car_x - car_width/2, car_y - car_height/2]]
        # ---------visualize----
        # cur = car.curr_state
        # car.curr_xy[0], car.curr_xy[1]= car_x, car_y
        # pts = np.array([[car_s, car_n, 0, 0, 0]])
        # visualize(np.expand_dims(pts[0, :], 1), obstacle, car)
        # car.curr_xy[0], car.curr_xy[1]= cur[0], cur[1]
        # --------------------
        for i in range(4):
            for j in range(4):
                if intersect(obs_corners[i], obs_corners[(i+1) % 4], car_corners[j], car_corners[(j+1) % 4]):
                    return True
        com_car_tested[count, :] = np.array([car_s, car_n]).squeeze()
        count += 1


    test_thetas = [i*(theta/4) for i in range(1,5)]
    for t in test_thetas: #curve 2 section
        car_s, car_n = s2 - abs(rad*np.sin(t)), n2 + rad * (1 - np.cos(t))
        car_x, car_y = convert_curvilinear2xy(car_s, car_n)
        car_corners = [[car_x - car_width/2, car_y + car_height/2],
                       [car_x + car_width/2, car_y + car_height/2],
                       [car_x + car_width/2, car_y - car_height/2],
                       [car_x - car_width/2, car_y - car_height/2]]
        # ---------visualize----
        # cur = car.curr_state
        # car.curr_xy[0], car.curr_xy[1]= car_x, car_y
        # pts = np.array([[car_s, car_n, 0, 0, 0]])
        # visualize(np.expand_dims(pts[0, :], 1), obstacle, car)
        # car.curr_xy[0], car.curr_xy[1]= cur[0], cur[1]
        # --------------------
        for i in range(4):
            for j in range(4):
                if intersect(obs_corners[i], obs_corners[(i+1) % 4], car_corners[j], car_corners[(j+1) % 4]):
                    return True
        com_car_tested[count, :] = np.array([car_s, car_n]).squeeze()
        count += 1
    
    
    dist = (obstacle.s + obs_length) - s2
    linear_points = [i*(dist/7) for i in range(1, 8)]
    for s_increment in linear_points:  # linear section
        car_s, car_n = s2 + s_increment, n2
        car_x, car_y = convert_curvilinear2xy(car_s, car_n)
        car_corners = [[car_x - car_width/2, car_y + car_height/2],
                       [car_x + car_width/2, car_y + car_height/2],
                       [car_x + car_width/2, car_y - car_height/2],
                       [car_x - car_width/2, car_y - car_height/2]]
        # ---------visualize----
        # cur = car.curr_state
        # car.curr_xy[0], car.curr_xy[1]= car_x, car_y
        # pts = np.array([[car_s, car_n, 0, 0, 0]])
        # visualize(np.expand_dims(pts[0, :], 1), obstacle, car)
        # car.curr_xy[0], car.curr_xy[1]= cur[0], cur[1]
        # --------------------
        for i in range(4):
            for j in range(4):
                # print(np.array([obs_corners[i], obs_corners[(i+1)%4], car_corners[j], car_corners[(j+1)%4]]))
                if intersect(obs_corners[i], obs_corners[(i+1) % 4], car_corners[j], car_corners[(j+1) % 4]):
                    return True
        com_car_tested[count, :] = np.array([car_s, car_n]).squeeze()
        count += 1
    # visualizecollision(com_car_tested, obstacle, car, name="collision check sample")
    return False


def cost_func(rad, theta, obstacle, curr_s, curr_delta, car):
    w_d = 0 #1
    w_gap = 80
    w_r = 1 #5
    w_theta = 1 #10

    front_distance = (obstacle.s - obstacle.length/2) - \
        (car.curr_state[0] + car.length/2)
    penalty_theta = front_distance * theta
    exp_term = 0.1*((penalty_theta-10)**4 - 8*(penalty_theta-10)**2 + 1)
    theta_func = 100/(5 + np.exp(-exp_term))
    theta_cost = w_theta * theta_func
    # distance  theta   result
    # small     small   low cost  <-- problem want this to be high
    # small     large   med_cost
    # large     small   med_cost
    # large     large   high_chost

    d = 2*rad*(1-np.cos(theta))  # prefers smaller d
    gap = 0
    if curr_delta - d > obstacle.delta:
        gap = 19.5 - (obstacle.delta + obstacle.width/2)
    else:
        gap = obstacle.delta - obstacle.width/2

    print("gap: ", gap)

    car_width = 4.5
    print("gap cost: ", w_gap/(gap))
    if gap <= car_width + 1:
        return np.inf
    return theta_cost + w_r/rad + w_gap/gap 

def visualizecollision(points, obstacle, car, name=""):
    timestep = 0.25
    x_arr, y_arr = [], []
    fig, ax = plt.subplots()
    plt.scatter([0], [0], color='r')
    prev_x, prev_y = 0, 0
    def s_dot_solver(s_curr, s_prev):
        if s_curr-s_prev < -100: #define -100 as a threshold value
            return (s_curr+226.097-s_prev)/timestep
        return (s_curr-s_prev)/timestep                         
    n_dot_solver = lambda n_curr, n_prev: (n_curr-n_prev)/timestep
    step = 2 if name == "collision check sample" else 1
    for i in range(0, points.shape[0], step):
        x, y = convert_curvilinear2xy(points[i][0], points[i][1])
        x_arr.append(x)
        y_arr.append(y)

        if i == 0:
            ax.add_patch(Rectangle((x - (car.width/2), y - (car.length/2)), car.width, car.length))
        else:
            s_dot = s_dot_solver(points[i][0], points[i-1][0])
            n_dot = n_dot_solver(points[i][1], points[i-1][1])
            s = points[i][0]
            track_piece = get_track_piece(s)
            car_global_angle_ofset = 0
            if track_piece == 0:
                car_global_angle_ofset = 0
            elif track_piece == 1:
                car_global_angle_ofset = s/18.0
            elif track_piece == 2:
                car_global_angle_ofset = s/18.0
            elif track_piece == 3:
                car_global_angle_ofset = np.pi 
            elif track_piece == 4 or track_piece == 5:
                car_global_angle_ofset = (s - 56.5)/18.0
            
            print(car_global_angle_ofset, track_piece)
            phi = np.arctan2(n_dot, s_dot)
            deg = (-phi + car_global_angle_ofset)*180/np.pi
            point_rot = np.array([x-car.width/2, y-car.length/2])
            ax.add_patch(Rectangle(point_rot, width=car.width, height=car.length, color=str(0.5), alpha=0.9, transform=Affine2D().rotate_deg_around(x, y, deg)+ax.transData))
            

    ax.add_patch(Rectangle((obstacle.x - (obstacle.width/2), obstacle.y - (obstacle.length/2)), obstacle.width, obstacle.length))
    
    ir, otr, straight = 18, 22.5 + 18, 56.5
    #--Draw inner top
    x = np.linspace(-ir + -ir, ir + -ir)
    y = np.sqrt(ir**2 - ((x + ir)) ** 2)
    ax.plot(x, y, c='orange')
    #--
    #--Draw outer top
    x = np.linspace((-otr - ir), (otr - ir))
    y = np.sqrt((otr)**2 - ((x + ir)) ** 2)
    ax.plot(x, y, c='orange')
    #--

    #--Draw inner bottom
    x = np.linspace(-ir + -ir, ir + -ir)
    y = -np.sqrt(ir**2 - ((x + ir)) ** 2) - straight
    ax.plot(x, y, c='orange')
    #--
    #--Draw outer bottom
    x = np.linspace((-otr - ir), (otr - ir))
    y = -np.sqrt((otr)**2 - ((x + ir)) ** 2) - straight
    ax.plot(x, y, c='orange')
    #--
    y = np.linspace(-straight, 0)
    x = y * 0 + 0
    ax.plot(x, y, c='orange')
    y = np.linspace(-straight, 0)
    x = y * 0 + (otr - ir)
    ax.plot(x, y, c='orange')
    y = np.linspace(-straight, 0)
    x = y * 0 + (- 2 * ir)
    ax.plot(x, y, c='orange')
    y = np.linspace(-straight, 0)
    x = y * 0 + (- ir - otr)
    ax.plot(x, y, c='orange')

    plt.title(name)
    plt.scatter(x_arr[0], y_arr[0], c='g', zorder = 2)
    plt.scatter(x_arr[1:], y_arr[1:], c='k', zorder=2)
    plt.axis('square')
    # plt.savefig("optimal_traj_%f.png"%time.time())
    plt.show()

    return


#SAMPLING
def sampling_method(car, obstacle): 
    #return optimal trajectory from samples, #change function to sample theta, calculate desired offset
    curr_s, curr_delta = car.curr_state[0], car.curr_state[1]
    R_min, R_max = 5, 15  # cm
    theta_min, theta_max = np.pi/8, np.pi/2
    sample_size = 40
    
    costs = np.zeros(sample_size)
    samples = []
    car_collision_positions = []

    left = np.random.randint(5, sample_size)

    for s in range(left):
        curr_R = np.random.uniform(R_min, R_max, 1)
        curr_theta = np.random.uniform(theta_min, theta_max, 1)
        desired_offset = curr_delta - curr_R * (1 - np.cos(curr_theta))
        samples.append(tuple((curr_R, curr_theta)))
        boolean_collide = if_collision(curr_R, curr_theta, obstacle, car)
        if boolean_collide:
            costs[s] = np.inf
            print("Collision")
            s = s + 1
            continue
        costs[s] = cost_func(curr_R, curr_theta, obstacle, curr_s, curr_delta, car)
        print("s: ", s)
        print("cost: ", costs[s])

    for s in range(sample_size - left):
        curr_R = np.random.uniform(-R_max, -R_min, 1)
        curr_theta = np.random.uniform(theta_min, theta_max, 1)
        desired_offset = curr_delta - curr_R * (1 - np.cos(curr_theta))
        samples.append(tuple((curr_R, curr_theta)))
        boolean_collide = if_collision(curr_R, curr_theta, obstacle, car)
        if boolean_collide:
            costs[s + left] = np.inf
            print("Collision")
            s = s + 1
            continue
        costs[s + left] = cost_func(curr_R, curr_theta, obstacle, curr_s, curr_delta, car)
        print("s: ", s + left)
        print("cost: ", costs[s+left])
        
    # print("Collided car positions: ", car_collision_positions)
    print(costs)
    idx = np.argmin(costs)
    best_r, best_theta = samples[idx]
    return best_r, best_theta


def det_obstacle_avoided(obstacle, car):
    # determine if obstacle has been avoided
    if car.s > obstacle.s:
        return True
    return False

# TRAJECTORY OPTIMATION


def get_A_and_B():
    # regression on dynamics model to get the A and B of dynamics
    # states: s, n, s_dot, n_dot, phi
    A = np.array([[1, 0, 0.25, 0, 0],
                  [0, 1, 0, 0.25, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])  # np.load("../../../savedAMatrix.npy")
    B = np.load("savedBmatrix2.npy")
    return A, B


def calc_input(car, reference_pts, A, B, num_of_horizon, obstacle):
    xdim = 5 #[s, ey, s_dot, n_dot, phi] 
    udim = 2 #[offset, accel]
    matrix_A, matrix_B = A, B
    matrix_Q = np.eye(5) * 100#Need to tune 
    matrix_R = np.zeros((2, 2)) #Need to tune
    matrix_Q[1][1] = 1500
    matrix_Q[0][0] = 500
    #get value of xt [4 x horizon +1]

    xt = None  # car.curr_state #[s, ey, v]

    start_timer = datetime.datetime.now()  # for debugging
    opti = ca.Opti()
    # define variables
    xvar = opti.variable(xdim, num_of_horizon + 1)  # total with dim 3
    # [offset, accel] --> use speed to call API
    uvar = opti.variable(udim, num_of_horizon)
    cost = 0

    x = reference_pts[:, 0] #determine the initial condition

    # initial condtion (in terms of r theta coords most likely)
    opti.subject_to(xvar[:, 0] == x)
    # dynamics + state/input constraints
    for i in range(num_of_horizon):  # horizon = 5
        # system dynamics
        opti.subject_to(
            xvar[:, i + 1] == ca.mtimes(matrix_A, xvar[:, i]) +
            ca.mtimes(matrix_B, uvar[:, i])
        )
        # min and max of offset
        opti.subject_to(-88 <= uvar[0, i])
        opti.subject_to(uvar[0, i] <= 88)

        # min and max of accel
        opti.subject_to(-200 <= uvar[1, i])
        opti.subject_to(uvar[1, i] <= 200)

        # input cost
        # put weight over speed and offset
        cost += ca.mtimes(uvar[:, i].T, ca.mtimes(matrix_R, uvar[:, i]))
    for i in range(num_of_horizon + 1):
        # min and max of ey
        # state cost
        xt = reference_pts[:, i]
        # tracking cost
        cost += ca.mtimes((xvar[:, i] - xt).T,
                          ca.mtimes(matrix_Q, xvar[:, i] - xt))
    # setup solver
    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti.minimize(cost)
    opti.solver("ipopt", option)
    sol = opti.solve()
    end_timer = datetime.datetime.now()
    solver_time = (end_timer - start_timer).total_seconds()
    print("solver time: {}".format(solver_time))
    # smoothing to stop jumping behavior
    x_pred = sol.value(xvar).T
    u_pred = sol.value(uvar).T
    u = u_pred[0, :] #MPC
    return u

def pd_input(car, offset_d, obstacle):
    #takes in a car object and a derired offset, outputs Anki control input (OFFSET ONLY)
    #use constant speed control
    def obs_detector(obstacle):
        if car.curr_state[0] < obstacle.s:
            delta_s = obstacle.s-car.curr_state[0]
        else:
            # add a lap to the obstacle
            delta_s = obstacle.s + 226.097 - car.curr_state[0]
        return delta_s
    delta_s = obs_detector(obstacle)
    if delta_s < 100:
        Kp = 20 #set control gains to nonzero if within acceptable distance
        Kd = 10
    else:
        Kp = 0 #outside of range set gains to 0
        Kd = 0
    error = offset_d - car.curr_state[1]
    error_dot = -car.curr_state[3] #get offset velocity n_dot, equals error_dot
    return error*Kp + error_dot*Kd #PD Controller



def get_obs(car_obj):
    obstacles = car_obj.obstacles
    return obstacles


if __name__ == "__main__":
    # Start up of ros nodes
    rospy.wait_for_service('last_color_image')
    rospy.init_node('car_mask', anonymous=True, disable_signals=True)
    pub = rospy.Publisher('/car_position', CarPosition, queue_size=10)
    last_image_service = rospy.ServiceProxy('last_color_image', ImageSrv)
    pointFromPixel = PointFromPixel()

    # Image Calibration
    while not rospy.is_shutdown():
        try:
            ros_img_msg = last_image_service().image_data
            pointFromPixel.get_6_points(ros_img_msg)
            flag = True
            break
        except KeyboardInterrupt:
            print('Keyboard Interrupt, exiting')
            break
        # Catch if anything went wrong with the Image Service
        except rospy.ServiceException as e:
            print("image_process: Service call failed: %s" % e)

    if flag:
        while not rospy.is_shutdown():
            try:
                caraddr = "C9:08:8F:21:77:E3"
                car = Ankicar(caraddr)
                np.set_printoptions(suppress=True)
                getCarPos = car.getCarPos
                ros_img_msg = last_image_service().image_data

                car.get_smoothed_position(pointFromPixel, last_image_service)
                avoided = False
                timestep = 0.25  # TODO: set this correctly
                horizon = 3
                obstacle_detected = False
                A, B = get_A_and_B()
                obstacles = getCarPos.obstacles
                print(obstacles)
                #obstacle = obstacles[0]

                #obstacle = obstacles[0]
                do_MPC = False  # else do OL control
                do_P = False #do proportional control
                do_PD = True #do derivative control
                do_PD_setpoint = False #Setpoint PD control with trajectory optimization

                def control_func(getCarPos, obstacles, do_P, do_D):
                    #Takes inputs do_P, do_D, do_I - Equal 1 to ADD the term, equal 0 to IGNORE
                    #apply PD Control to overtaking
                    #u =  Kp * (nd - n) - Kd * n_dot (nd is constant)
                    car.set_speed(700)
                    Kp = 15*do_P  # open loop gain - converts from mm to cm
                    Kd = 5*do_D #derivative gain
                    # call function to take in obstacle and car and return new car position
                    car.get_smoothed_position(pointFromPixel, last_image_service)  # get car position
                    
                    def obs_detector(obstacle):
                        if car.curr_state[0] < obstacle.s:
                            delta_s = obstacle.s-car.curr_state[0]
                        else:
                            # add a lap to the obstacle
                            delta_s = obstacle.s + 226.097 - car.curr_state[0]
                        return delta_s
                    print(obstacles)
                    obstacle = obstacles[0]
                    print("Obstacle length: ", obstacle.length)
                    print("Obstacle delta: ", obstacle.delta)
                    obstacles_mapped = map(obs_detector, obstacles)
                    obstacles_sorted = np.argsort(obstacles_mapped)
                    if False:#len(obstacles_sorted) >= 2:
                        smallest = [obstacles[obstacles_sorted[0]], obstacles[obstacles_sorted[1]]] #Two smallest obstacles list
                    else:
                        smallest = obstacle

                    inside_obs_offset = smallest.delta
                    outside_obs_offset = 19.5-smallest.delta
                    delta_s = obs_detector(smallest) #get delta s

                    avoidance_thresh = (obstacle.length**2+obstacle.width**2)**0.5
                    print("Avoidance threshold: ", avoidance_thresh)
                    print("If cond 2", abs(obstacle.delta -car.curr_state[1]) < avoidance_thresh)

                    if delta_s < 75 and abs(obstacle.delta - car.curr_state[1]) < avoidance_thresh:
                        if inside_obs_offset > outside_obs_offset:
                            print("cond 1")
                            target_offset = obstacle.delta-avoidance_thresh
                            error = car.curr_state[1] - target_offset
                            u_prop = -Kp*(error+car.width/2)
                            u_deriv = -Kd*(car.curr_state[3]) #derivative term
                            print("Control input: ", u_prop + u_deriv)
                            car.set_offset(u_prop + u_deriv)
                        else:
                            print("cond 2")
                            target_offset = obstacle.delta+avoidance_thresh
                            error = -car.curr_state[1] + target_offset
                            u_prop = Kp*(error+car.width/2)
                            u_deriv = -Kd*(car.curr_state[3]) #derivative term
                            print("Control input: ", u_prop + u_deriv)
                            car.set_offset(u_prop + u_deriv)
                            
                while True:
                    car.set_speed(300)
                    #print(1)
                    ros_img_msg = last_image_service().image_data
                    #print(2)
                    obstacle_detected = getCarPos.get_lw_xy(ros_img_msg, pointFromPixel)  # get obstacle
                    obstacles = get_obs(getCarPos)
                    print(obstacles)
                    #print(3)
                    car.get_smoothed_position(pointFromPixel, last_image_service)
                    #print("Obstacle X position: ", obstacle.x)
                    if not obstacle_detected:
                        continue
                    print("obstacle_detected: ", obstacle_detected)
                    print(obstacle_detected and do_P)
                    if obstacle_detected and do_MPC:
                        avoided = False
                        while not avoided:
                            obstacles = get_obs(getCarPos)
                            obstacle = obstacles[0]
                            print("obstacle x:", obstacle.x)
                            start_time = time.time()
                            print("Time start:", start_time)
                            car.get_smoothed_position(
                                pointFromPixel, last_image_service)  # get car position

                            best_r, best_theta = sampling_method(car, obstacle)
                            print("Sampling Method: Best rad: " +
                                  str(best_r) + " best theta: " + str(best_theta))
                            # should be a [5, horizon+1] array which is best pts from sampling
                            reference_pts = best_traj_pts(
                                best_r, best_theta, horizon, car, timestep, obstacle)

                            # traj opt function
                            u_optimzed = calc_input(
                                car, reference_pts, A, B, horizon, obstacle)

                            # #MPC
                            offset = u_optimzed[0]
                            accel = u_optimzed[1]
                            print("accel", accel)
                            temp_speed = (accel * timestep +
                                          car.curr_state[2]) * 10
                            print("temp_speed", temp_speed)
                            speed = (temp_speed if temp_speed >
                                     300 else 300) if temp_speed < 800 else 800
                            print("speed", speed)

                            # call API for offset
                            # use accel to get speed then call API

                            off_val = offset-car.curr_state[1]

                            off_val = (off_val if off_val < -car.curr_state[1] else -car.curr_state[1]
                                       ) if off_val < 19.5-car.curr_state[1] else 19.5-car.curr_state[1]
                            offset = 10 * off_val
                            print("offset", -offset)
                            end_time = time.time()
                            print("End start:", end_time)
                            print("Total time: ", end_time - start_time)

                            car.set_offset(-offset)  # mm
                            car.set_speed(300)

                            avoided = det_obstacle_avoided(obstacle, car)
                        obstacle_detected = False
                    elif obstacle_detected and do_P:
                        print("Do P Control")
                        avoided = False
                        control_func(getCarPos, obstacles, 1, 0)
                    elif obstacle_detected and not do_P and do_PD:
                        print("PD Control")
                        #apply PD Control to overtaking
                        #u =  Kp * (nd - n) - Kd * n_dot (nd is constant)
                        avoided = False
                        control_func(getCarPos, obstacles, 1, 1)
                    elif obstacle_detected and do_PD_setpoint:
                        print("PD Setpoint Control")
                        #apply PD control with trajectory optimization points
                        avoided = False
                        while not avoided:
                            obstacles = get_obs(getCarPos)
                            obstacle = obstacles[0]
                            start_time = time.time()
                            print("Time start:", start_time)
                            car.get_smoothed_position(
                                pointFromPixel, last_image_service)  # get car position

                            best_r, best_theta = sampling_method(car, obstacle)
                            print("Sampling Method: Best rad: " +
                                  str(best_r) + " best theta: " + str(best_theta))
                            # should be a [5, horizon+1] array which is best pts from sampling
                            reference_pts = best_traj_pts(
                                best_r, best_theta, horizon, car, timestep, obstacle)

                            desired_state = reference_pts[1] #get final point state vector
                            desired_offset = desired_state[1] #get offset from state vector
                            print("Desired offset: ", desired_offset)
                            #call control input function
                            control_input = pd_input(car, desired_offset, obstacle)

                            end_time = time.time()
                            print("End start:", end_time)
                            print("Total time: ", end_time - start_time)

                            car.set_offset(control_input)  # mm
                            car.set_speed(300)

                            avoided = det_obstacle_avoided(obstacle, car)
                        obstacle_detected = False
                    else:
                        car.set_speed(300)
            except KeyboardInterrupt:
                print('Keyboard Interrupt, exiting')
                break

            # Catch if anything went wrong with the Image Service
            except rospy.ServiceException as e:
                print("image_process: Service call failed: %s" % e)
