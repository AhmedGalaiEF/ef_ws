#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "unitree_api/msg/request.hpp"
#include "unitree_api/msg/response.hpp"

class G1SportModeDriver : public rclcpp::Node
{
public:
    G1SportModeDriver() : Node("g1_sport_mode_driver")
    {
        // Create a publisher for the Unitree API requests
        request_publisher_ = this->create_publisher<unitree_api::msg::Request>("api/sport/request", 10);

        // Create a subscriber for the cmd_vel topic
        cmd_vel_subscriber_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10, std::bind(&G1SportModeDriver::cmd_vel_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "g1_sport_mode_driver started.");
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        // Create a new request message
        auto request_msg = std::make_shared<unitree_api::msg::Request>();

        // Set the request parameters
        request_msg->header.identity.api_id = 1004; // Sport mode API ID
        request_msg->parameter = "{\n    \"vx\": " + std::to_string(msg->linear.x) + ",\n    \"vy\": " + std::to_string(msg->linear.y) + ",\n    \"vyaw\": " + std::to_string(msg->angular.z) + "\n}";

        // Publish the request
        request_publisher_->publish(*request_msg);

        RCLCPP_INFO(this->get_logger(), "Published sport mode request: %s", request_msg->parameter.c_str());
    }

    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr request_publisher_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<G1SportModeDriver>());
    rclcpp::shutdown();
    return 0;
}
