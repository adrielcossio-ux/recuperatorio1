import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
import numpy as np
import time


class InverseKinematics(Node):

    def __init__(self):
        super().__init__('brazo_inverse_kinematics')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.target_sub = self.create_subscription(Point, 'target_position',
                                                   self.target_callback, 10)

        # Initial joint angles [q1, q2, q3, q4] in radians
        self.q = np.array([0.0, 0.0, 0.0, 0.0])

        # Robot link lengths (units must match target position units)
        self.l1 = 10
        self.l2 = 9
        self.l3 = 4
        self.l4 = 3

        self.timer = self.create_timer(0.1, self.update_joints)

        # Default target in Cartesian space [x, y, z].
        # WARNING: Your FK is planar (z=0). We'll force target z=0 in update.
        self.target_pos = np.array([1.5708, -0.785398, 0.785398])

        # IK parameters
        self.step_size = 0.05
        self.max_iterations = 100
        self.tolerance = 0.01
        # DLS damping factor (lambda); we will use lambda^2 in the formula
        self.damping_factor = 0.1

    def forward_kinematics(self, q):
        """Planar FK for a 4R arm; returns [x, y, z] with z=0."""
        q1, q2, q3, q4 = q

        x = (
            self.l1 * np.cos(q1)
            + self.l2 * np.cos(q1 + q2)
            + self.l3 * np.cos(q1 + q2 + q3)
            + self.l4 * np.cos(q1 + q2 + q3 + q4)
        )
        y = (
            self.l1 * np.sin(q1)
            + self.l2 * np.sin(q1 + q2)
            + self.l3 * np.sin(q1 + q2 + q3)
            + self.l4 * np.sin(q1 + q2 + q3 + q4)
        )
        z = 0.0

        return np.array([x, y, z])

    def jacobian(self, q):
        """Geometric Jacobian (positional part + a zero z-row). Shape 3x4."""
        q1, q2, q3, q4 = q

        j11 = -self.l1*np.sin(q1) - self.l2*np.sin(q1 + q2) - self.l3*np.sin(q1 + q2 + q3) - self.l4*np.sin(q1 + q2 + q3 + q4)
        j12 = -self.l2*np.sin(q1 + q2) - self.l3*np.sin(q1 + q2 + q3) - self.l4*np.sin(q1 + q2 + q3 + q4)
        j13 = -self.l3*np.sin(q1 + q2 + q3) - self.l4*np.sin(q1 + q2 + q3 + q4)
        j14 = -self.l4*np.sin(q1 + q2 + q3 + q4)

        j21 =  self.l1*np.cos(q1) + self.l2*np.cos(q1 + q2) + self.l3*np.cos(q1 + q2 + q3) + self.l4*np.cos(q1 + q2 + q3 + q4)
        j22 =  self.l2*np.cos(q1 + q2) + self.l3*np.cos(q1 + q2 + q3) + self.l4*np.cos(q1 + q2 + q3 + q4)
        j23 =  self.l3*np.cos(q1 + q2 + q3) + self.l4*np.cos(q1 + q2 + q3 + q4)
        j24 =  self.l4*np.cos(q1 + q2 + q3 + q4)

        # Planar arm => no z translation sensitivity in position rows
        j31 = 0.0
        j32 = 0.0
        j33 = 0.0
        j34 = 0.0

        return np.array([
            [j11, j12, j13, j14],
            [j21, j22, j23, j24],
            [j31, j32, j33, j34]
        ])

    def target_callback(self, msg: Point):
        """Receive target position; stored as [x, y, z]."""
        self.target_pos = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f"New target received: [{msg.x}, {msg.y}, {msg.z}]")

    def update_joints(self):
        """One IK iteration using Damped Least Squares on a rectangular Jacobian."""
        current_pos = self.forward_kinematics(self.q)

        # Force target z to 0 since FK returns z=0 for a planar arm
        target = self.target_pos.copy()
        target[2] = 0.0

        # Only XY error is meaningful here
        err_vec = target - current_pos
        err_xy = err_vec[:2]
        error_norm = np.linalg.norm(err_xy)

        self.get_logger().info(f"Current position: {current_pos}")
        self.get_logger().info(f"Target position:   {target}")
        self.get_logger().info(f"XY error norm:     {error_norm:.6f}")

        if error_norm > self.tolerance:
            # --- Build positional Jacobian (2x4) from the full 3x4 J ---
            J_full = self.jacobian(self.q)   # 3x4
            J_pos = J_full[0:2, :]           # 2x4 => rows for x and y

            # --- Singularity assessment with SVD (rectangular-safe) ---
            # Smallest singular value close to 0 => near singularity
            U, S, Vt = np.linalg.svd(J_pos, full_matrices=False)
            sigma_min = S[-1]
            self.get_logger().info(f"sigma_min(J_pos):  {sigma_min:.6e}")

            # --- Yoshikawa manipulability (using position-only Jacobian) ---
            # w = sqrt(det(J_pos * J_pos^T))  -> always >= 0
            JJt = J_pos @ J_pos.T            # 2x2
            w = float(np.sqrt(np.linalg.det(JJt)))
            self.get_logger().info(f"manipulability w:  {w:.6e}")

            # --- Damped Least Squares (rectangular-safe) ---
            # dq = (J^T J + λ^2 I)^-1 J^T e
            lam2 = (self.damping_factor ** 2)
            JtJ = J_pos.T @ J_pos            # 4x4
            rhs = J_pos.T @ err_xy            # 4x1
            dq = np.linalg.solve(JtJ + lam2 * np.eye(JtJ.shape[0]), rhs)

            # Joint update with step size
            self.q += self.step_size * dq

        # Publish updated joint states
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['q1', 'q2', 'q3', 'q4']
        msg.position = self.q.tolist()
        self.joint_pub.publish(msg)


def main():
    rclpy.init()
    node = InverseKinematics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

