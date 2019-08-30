from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
MAX_BRAKE = 400.0

KP = 0.1
KI = 0.1
KD = 0
MN = 0
MX = 0.2

TAU = 0.1
TS = 0.02

class TwistController(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, \
                       decel_limit, accel_limit, wheel_radius,  \
                       wheel_base, steer_ratio, max_lat_accel, \
                       max_steer_angle):

        self.vehicle_mass    = vehicle_mass   
        self.fuel_capacity   = fuel_capacity  
        self.brake_deadband  = brake_deadband  
        self.decel_limit     = decel_limit
        self.accel_limit     = accel_limit    
        self.wheel_radius    = wheel_radius   
        self.wheel_base      = wheel_base     
        self.steer_ratio     = steer_ratio    
        self.max_lat_accel   = max_lat_accel  
        self.max_steer_angle = max_steer_angle

        self.full_mass = self.vehicle_mass + self.fuel_capacity * GAS_DENSITY

        self.enabled = False

        self.yaw_controller = YawController(self.wheel_base, \
                                            self.steer_ratio, \
                                            0.1, \
                                            self.max_lat_accel, \
                                            self.max_steer_angle)
        self.lpf_vel = LowPassFilter(TAU, TS)
        self.pid = PID(kp=KP, ki=KI, kd=KD, mn=MN, mx=MX)

    def enable(self, enable):
        if enable and not self.enabled:
           self.enabled = True
           self.pid.reset()
        elif not enable and self.enabled:
           self.enabled = False
           self.pid.reset()
        else:
           pass

    def control(self, linear_vel, angular_vel, current_vel, dbw_enabled, dt):
        current_vel = self.lpf_vel.filt(current_vel)
        vel_error = linear_vel - current_vel
        throttle = self.pid.step(vel_error, dt)
        brake = 0

        if linear_vel==0. and current_vel< 0.1:
            throttle = 0
            brake = MAX_BRAKE
        elif throttle > 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.full_mass*self.wheel_radius
            brake = min(brake, MAX_BRAKE)

        steer = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        # Return throttle, brake, steer
        return throttle, brake, steer
