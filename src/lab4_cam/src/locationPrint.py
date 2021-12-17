from overdrive import Overdrive
import time

#def locationChangeCallback(addr, location, piece, speed, clockwise):
#    # Print out addr, piece ID, location ID of the vehicle, this print everytime when location changed
#    print("Location from " + addr + " : " + "Piece=" + str(piece) + " Location=" + str(location) + " Clockwise=" + str(clockwise))


car = Overdrive("C9:08:8F:21:77:E3")
#car.setLocationChangeCallback(locationChangeCallback) # Set location change callback to function above
car.changeSpeed(300, 1000) # Set car speed with speed = 500, acceleration = 1000
#car.changeLaneRight(1000, 1000) # Switch to next right lane with speed = 1000, acceleration = 1000
while True:
    print(car.piece)
    #car.changeLaneRight(1000, 1000)
    #car.changeLaneRight(1000, 1000)
    #time.sleep(1)
    car.changeLaneLeft(1000, 1000)
    car.changeLaneLeft(1000, 1000)
    time.sleep(1)
input() # Hold the program so it won't end abruptly