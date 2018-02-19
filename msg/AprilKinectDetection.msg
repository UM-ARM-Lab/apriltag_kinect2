# Message storing tag info and poses for apriltag_kinect2
Header header
uint32 id  # id of AprilTag (not necessarily unique)
float32 tag_size  # tag size in m
uint32 hammingDistance # The hamming distance between the detected code and the true code.
geometry_msgs/Pose pose  # pose of the marker in the camera frame
