### STEPS FOR WIFI CONNECTION ###


# get into this directory
sudo vim /etc/wpa_supplicant/wpa_supplicant-wlan0.conf

# add up the username and password in the file (replace YOUR_SSID and YOUR_WIFI_PASSWORD with your actual wifi credentials)

wpa_passphrase "YOUR_SSID" "YOUR_WIFI_PASSWORD" | sudo tee /etc/wpa_supplicant/wpa_supplicant-wlan0.conf >/dev/null



# run the following command to connect to the wifi network

sudo chmod 600 /etc/wpa_supplicant/wpa_supplicant-wlan0.conf

# run the service to connect to the wifi network

sudo systemctl daemon-reload
sudo systemctl enable rfkill-unblock-wifi.service
sudo systemctl enable wpa_supplicant@wlan0.service


# after running the above commands, you can check the status of the wifi connection using the following command:

sudo vim /etc/wpa_supplicant/wpa_supplicant-wlan0.conf : 
    
# make sure you have like this : 


network={
        ssid="KURZiot"
        psk="Un1tr33-26-deSR"
        #psk=42f3a70bed511a5e822ac5818e0481cd1d69a55a450365c2db9359e6ed3d4133
        scan_ssid=1
}

