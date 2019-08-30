#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
PUBLISHING_RATE = 10
LOG_SECONDS = 3
MAX_DECEL = .5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        self.waypoints_cb_last_time = rospy.get_time()
        self.final_waypoints_pub_last_time = rospy.get_time()
        self.name = self.__class__.__name__

        self.loop()

    def loop(self):
        rate = rospy.Rate(PUBLISHING_RATE)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                #rospy.logwarn("Entered publish waypoints.")
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        is_logging = False
        current_time = rospy.get_time()
        if (current_time - self.final_waypoints_pub_last_time) > LOG_SECONDS:
            self.final_waypoints_pub_last_time = current_time
            is_logging = True
        lane = Lane()
        lane.header = self.base_waypoints.header
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS

        if (is_logging):
            rospy.logwarn("%s: car @ %s", self.name, closest_idx)
            rospy.logwarn("%s: stop_idx[%s] < farthest_idx[%s] => %s", \
                            self.name,\
                            self.stopline_wp_idx, farthest_idx,\
                            self.stopline_wp_idx < farthest_idx)

        base_waypoints = self.base_waypoints.waypoints[closest_idx : farthest_idx]
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            if (is_logging):
                rospy.logwarn("%s: Deccelerating...",self.name)
                rospy.logwarn("%s: Base waypoints length: %s",self.name, len(base_waypoints))
            lane.waypoints = self.decelerate_waypoints(base_waypoints, self.stopline_wp_idx, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, stopline_wp_idx, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(stopline_wp_idx - closest_idx- 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2* MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
            #rospy.logwarn("KDTree created.")

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data
        current_time = rospy.get_time()
        if (current_time - self.waypoints_cb_last_time) > LOG_SECONDS:
           self.waypoints_cb_last_time = current_time
           rospy.logwarn("%s: Closest traffic stop idx received: %s",self.name, self.stopline_wp_idx)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    #def distance(self, waypoints, wp1, wp2):
    #    dist = 0
    #    dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
    #    for i in range(wp1, wp2+1):
    #        dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
    #        wp1 = i
    #    return dist

    def distance(self, waypoints, wp1, wp2):
        if wp1>=wp2:
            return 0
        dist = 0
        #start = np.min([wp1,wp2])
        #end = np.max([wp1,wp2])
        #if end==len(waypoints)-1:
        #    end = end - 1
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        for i in range(wp1, wp2):
            try:
                dist += dl(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
            except:
                rospy.logerr("%s: Error at i[%s], wp1[%s], wp2[%s], waypoints_len[%s]", \
                                self.name, i, wp1, wp2, len(waypoints))
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
