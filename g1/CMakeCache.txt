#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/asio.hpp>
#include <iomanip>
#include <thread> // Thread utilities used for delays

using namespace std;
using namespace boost::asio;

// Register map
const struct {
    const char* name;
    int address;
} regdict[] = {
    {"angleSet", 1486},
    {"forceSet", 1498},
    {"speedSet", 1522},
    {"angleAct", 1546},
    {"forceAct", 1582}
};

class SerialPort {
public:
    SerialPort(const string& port, unsigned int baudrate) : io(), serial(io, port) {
        serial.set_option(serial_port_base::baud_rate(baudrate));
        serial.set_option(serial_port_base::character_size(8)); // Data bits
        serial.set_option(serial_port_base::parity(serial_port_base::parity::none)); // No parity
        serial.set_option(serial_port_base::stop_bits(serial_port_base::stop_bits::one)); // One stop bit
        cout << "Serial port " << port << " opened" << endl;
    }

    void write(const vector<unsigned char>& data) {
        boost::asio::write(serial, buffer(data));
    }

    vector<unsigned char> read(size_t length) {
        vector<unsigned char> buffer(length);
        boost::asio::read(serial, boost::asio::buffer(buffer.data(), length));
        return buffer;  
    }

private:
    io_context io;
    serial_port serial;
};

// Resolve a register address by name
int getAddressByName(const string& reg_name) {
    for (const auto& reg : regdict) {
        if (reg.name == reg_name) {
            return reg.address;  // Return the matching address
        }
    }
    return -1;  // Return -1 if the name is not found
}

// Write-register helper
void write_register(SerialPort& serial, int address_decimal, const string& id_value, const vector<int>& values_to_write) {
    if (values_to_write.size() > 6) {
        cerr << "Too many values provided, at most 6 values can be written." << endl;
        return;
    }

    // Build and send the frame
    vector<unsigned char> send_buffer;
    send_buffer.push_back(0xEB); 
    send_buffer.push_back(0x90);
    send_buffer.push_back(static_cast<unsigned char>(stoi(id_value)));  // Convert the ID to unsigned char
    size_t num_values = values_to_write.size();
    unsigned char data_length = static_cast<unsigned char>(num_values * 2 + 3); 
    send_buffer.push_back(data_length);  
    send_buffer.push_back(0x12);  // Write-register command
    send_buffer.push_back(static_cast<unsigned char>(address_decimal & 0xFF));  // Address low byte
    send_buffer.push_back(static_cast<unsigned char>((address_decimal >> 8) & 0xFF));  // Address high byte

    // Append the values to be written
    for (int value : values_to_write) {
        send_buffer.push_back(static_cast<unsigned char>(value & 0xFF));  // Value low byte
        send_buffer.push_back(static_cast<unsigned char>((value >> 8) & 0xFF));  // Value high byte
    }

    // Compute the checksum starting from the ID field
    unsigned char checksum = 0;
    for (size_t k = 2; k < send_buffer.size(); ++k) {
        checksum += send_buffer[k];
    }

    send_buffer.push_back(checksum & 0xFF);  // Append checksum

    // Print the outgoing frame
    cout << "Sent write data: ";
    for (auto byte : send_buffer) {
        cout << hex << setw(2) << setfill('0') << (int)byte << " ";
    }
    cout << endl;

    // Send the frame
    serial.write(send_buffer);

    // Read and discard the write response
    try {
        vector<unsigned char> discard_buffer = serial.read(9); 
        cout << "Discarded response data after write, byte count: " << discard_buffer.size() << endl;
    } catch (const std::exception& e) {
        cerr << "Failed to read response data after write: " << e.what() << endl;
    }
}

// Read-register helper
void read_register(SerialPort& serial, int address_decimal, const string& id_value, size_t length_to_read) {
    vector<int> parsed_values;

    // Build and send the request frame
    vector<unsigned char> send_buffer;
    send_buffer.push_back(0xEB); 
    send_buffer.push_back(0x90);
    send_buffer.push_back(static_cast<unsigned char>(stoi(id_value)));  // Convert the ID to unsigned char
    send_buffer.push_back(0x04);
    send_buffer.push_back(0x11);
    send_buffer.push_back(static_cast<unsigned char>(address_decimal & 0xFF));  // Address low byte
    send_buffer.push_back(static_cast<unsigned char>((address_decimal >> 8) & 0xFF));  // Address high byte
    send_buffer.push_back(0x0C); // Register read length

    // Compute checksum
    unsigned char checksum = 0;
    for (size_t k = 2; k < send_buffer.size(); ++k) {
        checksum += send_buffer[k];
    }

    send_buffer.push_back(checksum & 0xFF);

    cout << "Sent full data: ";
    for (auto byte : send_buffer) {
        cout << hex << setw(2) << setfill('0') << (int)byte << " ";
    }
    cout << endl;

    serial.write(send_buffer);  // Send the frame through the serial port

    // Read the response frame
    vector<unsigned char> received_data = serial.read(length_to_read);

    // Print the raw response frame
    cout << "Received raw response data: ";
    for (auto byte : received_data) {
        cout << hex << setw(2) << setfill('0') << (int)byte << " ";
    }
    cout << endl;

    // Decode the response payload
    for (size_t j = 7; j < 19; j += 2) {
        if (j + 1 < received_data.size()) {
            int value = (received_data[j + 1] << 8) | received_data[j];
            if (value > 6000) {
                value = 0;
            }
            parsed_values.push_back(value);
        }
    }

    // Print parsed values
    cout << "Values: ";
    for (int value : parsed_values) {
        cout << dec << value << " ";
    }
    cout << endl;
}

// Demo entry point
int main() {
    string port = "/dev/ttyUSB0";  // Change this to the actual serial port name
    SerialPort serial(port, 115200);  // Open the serial port

    string id_value = "01";

    // Example write operations
    write_register(serial, getAddressByName("speedSet"), id_value, {1000, 1000, 1000, 1000, 1000, 1000});
    this_thread::sleep_for(std::chrono::seconds(1));

    write_register(serial, getAddressByName("angleSet"), id_value, {0, 1000, 1000, 1000, 1000, 0});
    this_thread::sleep_for(std::chrono::seconds(1));

    write_register(serial, getAddressByName("forceSet"), id_value, {1000, 1000, 1000, 1000, 1000, 1000});
    this_thread::sleep_for(std::chrono::seconds(1));

    write_register(serial, getAddressByName("angleSet"), id_value, {1000, 1000, 1000, 1000, 1000, 1000});
    this_thread::sleep_for(std::chrono::seconds(1));

    // Example read operations
    read_register(serial, getAddressByName("angleSet"), id_value, 20);
    this_thread::sleep_for(std::chrono::seconds(1));
    read_register(serial, getAddressByName("angleAct"), id_value, 20);
    this_thread::sleep_for(std::chrono::seconds(1));
    read_register(serial, getAddressByName("forceAct"), id_value, 20);
    this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}
