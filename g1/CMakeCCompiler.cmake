#include <iostream>
#include <bitset>
#include <sstream>
#include <string>
#include <vector>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <iomanip>
#include <thread>

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
    SerialPort(const string& port, unsigned int baudrate)
        : io(), serial(io, port), timer(io), timeout(false) {
        serial.set_option(serial_port_base::baud_rate(baudrate));
        serial.set_option(serial_port_base::character_size(8)); // Data bits
        serial.set_option(serial_port_base::parity(serial_port_base::parity::none)); // No parity
        serial.set_option(serial_port_base::stop_bits(serial_port_base::stop_bits::one)); // One stop bit
        cout << "Serial port " << port << " opened" << endl;
    }

    void write(const vector<unsigned char>& data) {
        boost::asio::write(serial, buffer(data));
    }

    vector<unsigned char> read(size_t length, int timeout_ms = 500) {
        vector<unsigned char> buffer(length);
        timeout = false;

        timer.expires_after(std::chrono::milliseconds(timeout_ms));
        timer.async_wait(boost::bind(&SerialPort::on_timeout, this, boost::asio::placeholders::error));

        boost::system::error_code ec;
        size_t bytes_transferred = 0;
        serial.async_read_some(boost::asio::buffer(buffer),
                               boost::bind(&SerialPort::on_read_complete, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred, &ec, &bytes_transferred));

        io.run();
        io.reset();

        if (timeout) {
            throw std::runtime_error("Serial port read timed out");
        }

        if (ec) {
            throw boost::system::system_error(ec);
        }

        buffer.resize(bytes_transferred);
        return buffer;
    }

private:
    void on_timeout(const boost::system::error_code& ec) {
        if (!ec) {
            timeout = true;
            serial.cancel(); 
        }
    }

    void on_read_complete(const boost::system::error_code& ec, size_t bytes_transferred, boost::system::error_code* out_ec, size_t* out_bytes_transferred) {
        *out_ec = ec;
        *out_bytes_transferred = bytes_transferred;
        timer.cancel(); 
    }

    io_context io;
    serial_port serial;
    steady_timer timer;
    bool timeout;
};

int getAddressByName(const string& reg_name) {
    for (const auto& reg : regdict) {
        if (reg.name == reg_name) {
            return reg.address;  
        }
    }
    return -1;  
}

// Address conversion helpers
string convertAddressToBinary_read(int address_dec, const string& id_value) {
    string address_bin = bitset<32>(address_dec).to_string();
    size_t first_one = address_bin.find('1');

    if (first_one != string::npos) {
        address_bin = address_bin.substr(first_one);
    } else {
        address_bin = "0"; 
    }

    string formatted_output = "0000000" + address_bin + "000000000000" + id_value;
    return formatted_output;
}

string convertAddressToBinary_write(int address_dec, const string& id_value) {
    string address_bin = bitset<32>(address_dec).to_string();
    size_t first_one = address_bin.find('1');

    if (first_one != string::npos) {
        address_bin = address_bin.substr(first_one);
    } else {
        address_bin = "0"; 
    }

    string formatted_output = "0000010" + address_bin + "000000000000" + id_value;
    return formatted_output;
}

string binaryToHex(const string& binary_string) {
    unsigned long decimal_value = stoul(binary_string, nullptr, 2);
    stringstream ss;
    ss << hex << uppercase << decimal_value;
    return ss.str();
}

// Convert hexadecimal text into protocol byte order
vector<unsigned char> processHexData(const string& hex_data) {
    string padded_hex = hex_data;
    if (padded_hex.length() < 8) {
        padded_hex = string(8 - padded_hex.length(), '0') + padded_hex;
    }

    vector<unsigned char> processed_data;

    for (size_t i = padded_hex.length(); i > 0; i -= 2) {
        string byte_str = padded_hex.substr(i - 2, 2);
        unsigned char byte_value = static_cast<unsigned char>(stoi(byte_str, nullptr, 16));
        processed_data.push_back(byte_value);
    }

    return processed_data;
}

void write_register(SerialPort& serial, int address_decimal, const string& id_value, const vector<int>& values_to_write) {
    if (values_to_write.size() > 6) {
        cerr << "Too many values provided, at most 6 values can be written." << endl;
        return;
    }

    vector<int> first_chunk(values_to_write.begin(), values_to_write.begin() + min(4, (int)values_to_write.size()));
    vector<int> second_chunk(values_to_write.begin() + first_chunk.size(), values_to_write.end());

    // Build and send the first frame
    auto construct_and_send = [&](const vector<int>& chunk, int addr) {
        // Convert the address from binary representation to hexadecimal bytes
        string result_bin = convertAddressToBinary_write(addr, id_value);
        string result_hex = binaryToHex(result_bin);
        vector<unsigned char> processed_data = processHexData(result_hex);

        // Build the data frame
        vector<unsigned char> send_buffer;
        send_buffer.push_back(0xAA);
        send_buffer.push_back(0xAA);
        send_buffer.insert(send_buffer.end(), processed_data.begin(), processed_data.end());

        // Append payload values
        for (int value : chunk) {
            send_buffer.push_back(value & 0xFF);  // Low byte
            send_buffer.push_back((value >> 8) & 0xFF);  // High byte
        }

        while (send_buffer.size() < 14) {
            send_buffer.push_back(0x00);
        }

        // Append fixed trailer bytes
        send_buffer.push_back(chunk.size() * 2);  // Payload length
        send_buffer.push_back(0x00);
        send_buffer.push_back(0x01);
        send_buffer.push_back(0x00);

        // Compute checksum
        unsigned char checksum = 0;
        for (size_t k = 2; k < send_buffer.size(); ++k) {
            checksum += send_buffer[k];
        }
        send_buffer.push_back(checksum & 0xFF);
        send_buffer.push_back(0x55);
        send_buffer.push_back(0x55);

        // Print and send the frame
        cout << "Sent full data: ";
        for (auto byte : send_buffer) {
            cout << hex << setw(2) << setfill('0') << (int)byte << " ";
        }
        cout << endl;

        serial.write(send_buffer);

        // Read and discard the response
        try {
            vector<unsigned char> discard_buffer = serial.read(32, 500); 
        } catch (const std::exception& e) {
            cerr << "Failed to read response data after write: " << e.what() << endl;
        }
    };

    // Send the first frame
    construct_and_send(first_chunk, address_decimal);

    // Send the second frame if there are remaining values
    if (!second_chunk.empty()) {
        construct_and_send(second_chunk, address_decimal + 8);  // Shift the address by 8
    }
}



void read_register(SerialPort& serial, int address_decimal, const string& id_value, size_t length_to_read) {
    vector<int> parsed_values1; // Parsed values from the first frame
    vector<int> parsed_values2; // Parsed values from the second frame

    vector<unsigned char> send_buffer; 

    string result_bin = convertAddressToBinary_read(address_decimal, id_value);
    string result_hex = binaryToHex(result_bin);  
    vector<unsigned char> processed_data = processHexData(result_hex);

    send_buffer.push_back(0xAA); 
    send_buffer.push_back(0xAA);
    send_buffer.insert(send_buffer.end(), processed_data.begin(), processed_data.end());
    send_buffer.push_back(0x08); 
    send_buffer.insert(send_buffer.end(), 7, 0x00); 
    send_buffer.push_back(0x01);
    send_buffer.push_back(0x00);
    send_buffer.push_back(0x01);
    send_buffer.push_back(0x00);

    unsigned char checksum = 0;
    for (size_t k = 2; k < send_buffer.size(); ++k) {
        checksum += send_buffer[k];
    }
    send_buffer.push_back(checksum & 0xFF); // Append checksum
    send_buffer.push_back(0x55);
    send_buffer.push_back(0x55);

    // Print the outgoing frame
    cout << "Sent full data: ";
    for (auto byte : send_buffer) {
        cout << hex << setw(2) << setfill('0') << (int)byte << " ";
    }
    cout << endl;

    serial.write(send_buffer); 
    // Read the response frame
    vector<unsigned char> received_data = serial.read(length_to_read);

    // Print the raw response frame
    cout << "Received raw data: ";
    for (auto byte : received_data) {
        cout << hex << setw(2) << setfill('0') << (int)byte << " ";
    }
    cout << endl;

    // Find the first 0xAA marker
    size_t start_pos = 0;
    while (start_pos < received_data.size() && received_data[start_pos] != 0xAA) {
        start_pos++;
    }

    // Start decoding from the first 0xAA marker
    for (size_t j = start_pos + 6; j < start_pos + 14; j += 2) {
        if (j + 1 < received_data.size()) {
            int value = (received_data[j + 1] << 8) | received_data[j];
            if (value > 6000) { 
                value = 0;
            }
            parsed_values1.push_back(value);
        }
    }

    address_decimal += 8;  // Shift address forward
    result_bin = convertAddressToBinary_read(address_decimal, id_value);
    result_hex = binaryToHex(result_bin);  // Convert the binary string to hexadecimal
    processed_data = processHexData(result_hex);


    send_buffer.push_back(0xAA); 
    send_buffer.push_back(0xAA);
    send_buffer.insert(send_buffer.end(), processed_data.begin(), processed_data.end());
    send_buffer.push_back(0x04); 
    send_buffer.insert(send_buffer.end(), 7, 0x00); 
    send_buffer.push_back(0x01);
    send_buffer.push_back(0x00);
    send_buffer.push_back(0x01);
    send_buffer.push_back(0x00);

    checksum = 0;
    for (size_t k = 2; k < send_buffer.size(); ++k) {
        checksum += send_buffer[k];
    }
    send_buffer.push_back(checksum & 0xFF); // Append checksum
    send_buffer.push_back(0x55);
    send_buffer.push_back(0x55);

    cout << "Sent full data (second half): ";
    for (auto byte : send_buffer) {
        cout << hex << setw(2) << setfill('0') << (int)byte << " ";
    }
    cout << endl;

    serial.write(send_buffer); 

    received_data = serial.read(length_to_read);

    // Print the raw response from the second frame
    cout << "Received raw data (second half): ";
    for (auto byte : received_data) {
        cout << hex << setw(2) << setfill('0') << (int)byte << " ";
    }
    cout << endl;

    // Find the first 0xAA marker
    start_pos = 0;
    while (start_pos < received_data.size() && received_data[start_pos] != 0xAA) {
        start_pos++;
    }

    // Start decoding from the first 0xAA marker
    for (size_t j = start_pos + 6; j < start_pos + 14; j += 2) {
        if (j + 1 < received_data.size()) {
            int value = (received_data[j + 1] << 8) | received_data[j];
            if (value > 6000) { 
                value = 0;
            }
            parsed_values2.push_back(value);
        }
    }

    // Print the parsed output
    cout << "Values: ";
    // Print values parsed from the first frame
    for (int value : parsed_values1) {
        cout << dec << value << " ";
    }

    // Print the first two values parsed from the second frame
    for (size_t i = 0; i < 2 && i < parsed_values2.size(); ++i) {
        cout << dec << parsed_values2[i] << " ";
    }
    cout << endl;
}

int main() {
    string port = "/dev/ttyUSB1";  // Change this to the actual serial port name
    SerialPort serial(port, 115200);

    string id_value = "01";

    // Example write operations
    write_register(serial, getAddressByName("speedSet"), id_value, {1000, 1000, 1000, 1000, 1000, 1000});
    this_thread::sleep_for(std::chrono::seconds(1));

    write_register(serial, getAddressByName("angleSet"), id_value, {1000, 1000, 1000, 1000, 1000, 0});
    this_thread::sleep_for(std::chrono::seconds(1));

    write_register(serial, getAddressByName("forceSet"), id_value, {1000, 1000, 1000, 1000, 1000, 1000});
    this_thread::sleep_for(std::chrono::seconds(1));

    write_register(serial, getAddressByName("angleSet"), id_value, {1000, 1000, 1000, 1000, 1000, 1000});
    this_thread::sleep_for(std::chrono::seconds(1));

    read_register(serial, getAddressByName("angleSet"), id_value, 23);
    this_thread::sleep_for(std::chrono::seconds(1));
    read_register(serial, getAddressByName("angleAct"), id_value, 23);
    this_thread::sleep_for(std::chrono::seconds(1));
    read_register(serial, getAddressByName("forceAct"), id_value, 23);
    this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}
