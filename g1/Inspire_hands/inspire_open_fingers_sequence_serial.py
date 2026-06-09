#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/types.h>
#include <termios.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/select.h>
#include <time.h>
#include <pthread.h>
#include "hand_api.h"

// int pthread_create(
//     pthread_t *restrict thread,
//     const pthread_attr_t *restrict attr,
//     void *(*start_routine)(void*),
//     void *restrict arg
// );

ActuatorStatusTypeDef tHand;
UartDataBuff tUartData = {0};

volatile int fd_left, fd_right, nByte;
volatile uint8_t sum, g_unHand_id = 0x01; // Hand ID
volatile uint16_t g_unNum = 50;			  // Define data length

void *Action(int numOfAction)
{
	switch (numOfAction)
	{
	case 0:
		Write_Hand_Angle_Position(0x01, 1000, 1000, 1000, 1000, 1000, 1000);

		write(fd_left, tUartData.m_send_array, tUartData.m_tx_len);
		write(fd_right, tUartData.m_send_array, tUartData.m_tx_len);
			printf("Sent command frame: ");
		usleep(500000);

               Read_Hand_Data(0x01,1546,0x0c); 
		write(fd_left, tUartData.m_send_array, tUartData.m_tx_len);
		write(fd_right, tUartData.m_send_array, tUartData.m_tx_len);
			printf("Sent command frame: ");
		for (int i = 0; i < tUartData.m_tx_len; i++)
		{
			printf("%02x ", tUartData.m_send_array[i]);
		}
		printf("\n");
		usleep(500000);
		
		Write_Hand_Angle_Position(0x01, 200, 200, 200, 200, 400, 600);
		write(fd_left, tUartData.m_send_array, tUartData.m_tx_len);
		write(fd_right, tUartData.m_send_array, tUartData.m_tx_len);
			printf("Sent command frame: ");
		for (int i = 0; i < tUartData.m_tx_len; i++)
		{
			printf("%02x ", tUartData.m_send_array[i]);
		}
		printf("\n");
		usleep(500000);
		
               Read_Hand_Data(0x01,1546,0x0c); 
		write(fd_left, tUartData.m_send_array, tUartData.m_tx_len);
		write(fd_right, tUartData.m_send_array, tUartData.m_tx_len);
			printf("Sent command frame: ");
		for (int i = 0; i < tUartData.m_tx_len; i++)
		{
			printf("%02x ", tUartData.m_send_array[i]);
		}
		printf("\n");
		usleep(500000);
		
		break;

	default:
	    printf("NO ACTION\n");
		break;
	}
}

void *ThreadEntry_left(void *arg)
{
	(void)arg;
	int ch = 0, i = 0, j = 0;
	while (1)
	{
		while ((nByte = read(fd_left, &ch, 1)) > 0)
		{
			tUartData.m_rec_array[i] = ch;
			if ((i > 1) && (i < (tUartData.m_rec_array[3] + 4)))
			{
				sum += tUartData.m_rec_array[i];
			}
			i++;
			j = 1;

			nByte = 0;
		}
if ((j == 1) && (sum == tUartData.m_rec_array[i - 1])) {
            // Print the raw response frame
	            printf("Raw response data: ");
            for (int k = 0; k < i; k++) {
                printf("%02x ", tUartData.m_rec_array[k]); // Print as hexadecimal
            }
            printf("\n");

    i = 0;
    j = 0;
    sum = 0;
    Uart_Rece_Data_Parsing();
    memset(tUartData.m_rec_array, 0, sizeof(*tUartData.m_rec_array));

    printf("Left LittleFinger.m_current_angle %hd\n", tHand.tLittleFinger.m_current_angle);    
    printf("Left RingFinger.m_current_angle %hd\n", tHand.tRingFinger.m_current_angle);
    printf("Left MiddleFinger.m_current_angle %hd\n", tHand.tMiddleFinger.m_current_angle);  
    printf("Left IndexFinger.m_current_angle %hd\n", tHand.tIndexFinger.m_current_angle);     
    printf("Left ThumbSide.m_current_angle %hd\n", tHand.tThumbSide.m_current_angle);
    printf("Left ThumbThumbBend.m_current_angle %hd\n", tHand.tThumbBend.m_current_angle);    
    }
 }
}

void *ThreadEntry_right(void *arg)
{
	(void)arg;
	int ch = 0, i = 0, j = 0;
	while (1)
	{
		while ((nByte = read(fd_right, &ch, 1)) > 0)
		{
			tUartData.m_rec_array[i] = ch;
			if ((i > 1) && (i < (tUartData.m_rec_array[3] + 4)))
			{
				sum += tUartData.m_rec_array[i];
			}
			i++;
			j = 1;

			nByte = 0;
		}
		if ((j == 1) && (sum == tUartData.m_rec_array[i - 1]))
		{
			i = 0;
			j = 0;
			sum = 0;
			Uart_Rece_Data_Parsing();
			memset(tUartData.m_rec_array, 0, sizeof(*tUartData.m_rec_array));
			printf("%d\n", j);
			printf("%d\n", tHand.tThumbSide.m_current_angle);
		}
		// memset(tUartData.m_rec_array,0,sizeof(*tUartData.m_rec_array));
		usleep(50000);
	}
}
/*seconds: the seconds; mseconds: the micro seconds*/
void setTimer(int seconds, int mseconds, int com)
{
	struct timeval temp;

	temp.tv_sec = seconds;
	temp.tv_usec = mseconds * 1000;

	select(0, NULL, NULL, NULL, &temp);

	read(com, tUartData.m_rec_array, sizeof(*tUartData.m_rec_array));
	printf("%x\n", *tUartData.m_rec_array);
	memset(tUartData.m_rec_array, 0, sizeof(*tUartData.m_rec_array));

	return;
}
int set_opt(int fd, int nSpeed, int nBits, char nEvent, int nStop)
{
	struct termios newtio, oldtio;
	if (tcgetattr(fd, &oldtio) != 0)
	{
		perror("SetupSerial 1");
		return -1;
	}
	bzero(&newtio, sizeof(newtio));
	newtio.c_cflag |= CLOCAL | CREAD;
	newtio.c_cflag &= ~CSIZE;

	switch (nBits)
	{
	case 7:
		newtio.c_cflag |= CS7;
		break;
	case 8:
		newtio.c_cflag |= CS8;
		break;
	}

	switch (nEvent)
	{
	case 'O':
		newtio.c_cflag |= PARENB;
		newtio.c_cflag |= PARODD;
		newtio.c_iflag |= (INPCK | ISTRIP);
		break;
	case 'E':
		newtio.c_iflag |= (INPCK | ISTRIP);
		newtio.c_cflag |= PARENB;
		newtio.c_cflag &= ~PARODD;
		break;
	case 'N':
		newtio.c_cflag &= ~PARENB;
		break;
	}

	switch (nSpeed)
	{
	case 2400:
		cfsetispeed(&newtio, B2400);
		cfsetospeed(&newtio, B2400);
		break;
	case 4800:
		cfsetispeed(&newtio, B4800);
		cfsetospeed(&newtio, B4800);
		break;
	case 9600:
		cfsetispeed(&newtio, B9600);
		cfsetospeed(&newtio, B9600);
		break;
	case 115200:
		cfsetispeed(&newtio, B115200);
		cfsetospeed(&newtio, B115200);
		break;
	case 460800:
		cfsetispeed(&newtio, B460800);
		cfsetospeed(&newtio, B460800);
		break;
	default:
		cfsetispeed(&newtio, B9600);
		cfsetospeed(&newtio, B9600);
		break;
	}
	if (nStop == 1)
		newtio.c_cflag &= ~CSTOPB;
	else if (nStop == 2)
		newtio.c_cflag |= CSTOPB;
	newtio.c_cc[VTIME] = 0;
	newtio.c_cc[VMIN] = 0;
	tcflush(fd, TCIFLUSH);
	if ((tcsetattr(fd, TCSANOW, &newtio)) != 0)
	{
		perror("com set error");
		return -1;
	}

	printf("Serial port setup completed.\n\r");
	return 0;
}

int Read_Hand_Data(uint8_t ID, uint16_t adress, uint16_t len)
{
	u_int8_t i = 0;
	u_int8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x04;
	tUartData.m_send_array[4] = 0x11;
	tUartData.m_send_array[5] = adress & 0xFF;      // Low byte
	tUartData.m_send_array[6] = (adress >> 8) & 0xFF; // High byte
        tUartData.m_send_array[7] = len;
	for (i = 2; i < 8; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[8] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 9;
	
            // Print the raw transmit frame
    printf("Built raw serial input frame: ");
    for (i = 0; i < tUartData.m_tx_len; i++)
    {
        printf("%02x ", tUartData.m_send_array[i]); // Print as hexadecimal
    }
    printf("\n");
	
	return 0;
}
int Read_Hand_Register_Data(uint8_t ID, uint16_t val, uint16_t vallen)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x04;
	tUartData.m_send_array[4] = kCmd_Handg3_Read;
	tUartData.m_send_array[5] = val & 0xFF;
	tUartData.m_send_array[6] = val >> 8;
	tUartData.m_send_array[7] = vallen;
	for (i = 2; i < 8; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[8] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 9;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Save_Hand_Data(uint8_t ID)
** Descriptions:        Save hand data
** input parameters:    Input ID of the hand
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/

int Save_Hand_Data(uint8_t ID)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x04;
	tUartData.m_send_array[4] = kCmd_Handg3_Write;
	tUartData.m_send_array[5] = 0xED;
	tUartData.m_send_array[6] = 0x03;
	tUartData.m_send_array[7] = 0x01;
	for (i = 2; i < 8; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[8] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 9;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Change_Hand_ID(uint8_t ID,uint8_t val)
** Descriptions:        Change hand ID
** input parameters:    ID:Input ID of the drive
						val:Input another ID
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/

int Change_Hand_ID(uint8_t ID, uint8_t val)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x04;
	tUartData.m_send_array[4] = kCmd_Handg3_Write;
	tUartData.m_send_array[5] = 0xE8;
	tUartData.m_send_array[6] = 0x03;
	tUartData.m_send_array[7] = val;
	for (i = 2; i < 8; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[8] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 9;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Change_Hand_Baud(uint8_t ID,uint8_t baud)
** Descriptions:        Change baud of the hand
** input parameters:    ID:ID of the current hand
						baud:Representative number of baud rate,0--19200 1--57600 2--115200
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/

int Change_Hand_Baud(uint8_t ID, uint8_t baud)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x04;
	tUartData.m_send_array[4] = kCmd_Handg3_Write;
	tUartData.m_send_array[5] = 0xE9;
	tUartData.m_send_array[6] = 0x03;
	tUartData.m_send_array[7] = baud;
	for (i = 2; i < 8; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[8] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 9;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Clear_Hand_Error(uint8_t ID)
** Descriptions:        Clear drive error
** input parameters:    Input ID of the drive
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/

int Clear_Hand_Error(uint8_t ID)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x04;
	tUartData.m_send_array[4] = kCmd_Handg3_Write;
	tUartData.m_send_array[5] = 0xEC;
	tUartData.m_send_array[6] = 0x03;
	tUartData.m_send_array[7] = 0x01;
	for (i = 2; i < 8; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[8] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 9;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Reset_Hand(uint8_t ID)
** Descriptions:        Reset drive
** input parameters:    Input ID of the drive
** output parameters:   Outout command array
** Returned value:      No
*********************************************************************************************************/

int Reset_Hand(uint8_t ID)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x04;
	tUartData.m_send_array[4] = kCmd_Handg3_Write;
	tUartData.m_send_array[5] = 0xEE;
	tUartData.m_send_array[6] = 0x03;
	tUartData.m_send_array[7] = 0x01;
	for (i = 2; i < 8; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[8] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 9;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Force_Sensor_Calibration(uint8_t ID)
** Descriptions:        Calibrate the force sensor
** input parameters:    Input ID of the drive
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/

int Force_Sensor_Calibration(uint8_t ID)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x04;
	tUartData.m_send_array[4] = kCmd_Handg3_Write;
	tUartData.m_send_array[5] = 0xF1;
	tUartData.m_send_array[6] = 0x03;
	tUartData.m_send_array[7] = 0x01;
	for (i = 2; i < 8; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[8] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 9;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Write_Hand_Angle(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6)
** Descriptions:        Set the angle of each finger
** input parameters:    ID:Input ID of the drive
						val1:Little figure angle setting value
						val2:Ring finger angle setting value
						val3:Middle finger angle setting value
						val4:Index finger angle setting value
						val5:Thumb angle setting value
						val6:Thumb swing angle setting value
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/

int Write_Hand_Angle(uint8_t ID, uint16_t val1, uint16_t val2, uint16_t val3, uint16_t val4, uint16_t val5, uint16_t val6)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x0D;
	tUartData.m_send_array[4] = kCmd_Mc_All;
	tUartData.m_send_array[5] = val1 & 0xFF;
	tUartData.m_send_array[6] = val1 >> 8;
	tUartData.m_send_array[7] = val2 & 0xFF;
	tUartData.m_send_array[8] = val2 >> 8;
	tUartData.m_send_array[9] = val3 & 0xFF;
	tUartData.m_send_array[10] = val3 >> 8;
	tUartData.m_send_array[11] = val4 & 0xFF;
	tUartData.m_send_array[12] = val4 >> 8;
	tUartData.m_send_array[13] = val5 & 0xFF;
	tUartData.m_send_array[14] = val5 >> 8;
	tUartData.m_send_array[15] = val6 & 0xFF;
	tUartData.m_send_array[16] = val6 >> 8;
	for (i = 2; i < 17; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[17] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 18;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Write_Hand_Drive_Position(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6)
** Descriptions:        Through register set the position of each finger
** input parameters:    ID:Input ID of the drive
						val1:Little figure position setting value
						val2:Ring finger position setting value
						val3:Middle finger position setting value
						val4:Index finger position setting value
						val5:Thumb angle position value
						val6:Thumb swing position setting value
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/

int Write_Hand_Drive_Position(uint8_t ID, uint16_t val1, uint16_t val2, uint16_t val3, uint16_t val4, uint16_t val5, uint16_t val6)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x0F;
	tUartData.m_send_array[4] = kCmd_Handg3_Write;
	tUartData.m_send_array[5] = 0xC2;
	tUartData.m_send_array[6] = 0x05;
	tUartData.m_send_array[7] = val1 & 0xFF;
	tUartData.m_send_array[8] = val1 >> 8;
	tUartData.m_send_array[9] = val2 & 0xFF;
	tUartData.m_send_array[10] = val2 >> 8;
	tUartData.m_send_array[11] = val3 & 0xFF;
	tUartData.m_send_array[12] = val3 >> 8;
	tUartData.m_send_array[13] = val4 & 0xFF;
	tUartData.m_send_array[14] = val4 >> 8;
	tUartData.m_send_array[15] = val5 & 0xFF;
	tUartData.m_send_array[16] = val5 >> 8;
	tUartData.m_send_array[17] = val6 & 0xFF;
	tUartData.m_send_array[18] = val6 >> 8;
	for (i = 2; i < 19; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[19] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 20;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Write_Hand_Angle_Position(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6)
** Descriptions:        Through register set the Angle of each finger
** input parameters:    ID:Input ID of the drive
						val1:Little figure angle setting value
						val2:Ring finger angle setting value
						val3:Middle finger angle setting value
						val4:Index finger angle setting value
						val5:Thumb angle setting value
						val6:Thumb swing angle setting value
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/

int Write_Hand_Angle_Position(uint8_t ID, uint16_t val1, uint16_t val2, uint16_t val3, uint16_t val4, uint16_t val5, uint16_t val6)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x0F;
	tUartData.m_send_array[4] = kCmd_Handg3_Write;
	tUartData.m_send_array[5] = 0xCE;
	tUartData.m_send_array[6] = 0x05;
	tUartData.m_send_array[7] = val1 & 0xFF;
	tUartData.m_send_array[8] = val1 >> 8;
	tUartData.m_send_array[9] = val2 & 0xFF;
	tUartData.m_send_array[10] = val2 >> 8;
	tUartData.m_send_array[11] = val3 & 0xFF;
	tUartData.m_send_array[12] = val3 >> 8;
	tUartData.m_send_array[13] = val4 & 0xFF;
	tUartData.m_send_array[14] = val4 >> 8;
	tUartData.m_send_array[15] = val5 & 0xFF;
	tUartData.m_send_array[16] = val5 >> 8;
	tUartData.m_send_array[17] = val6 & 0xFF;
	tUartData.m_send_array[18] = val6 >> 8;
	for (i = 2; i < 19; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[19] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 20;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Write_Hand_Force_Threshold(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6)
** Descriptions:        Through register set the force threshole of each finger
** input parameters:    ID:Input ID of the drive
						val1:Little figure force threshole setting value
						val2:Ring finger force threshole setting value
						val3:Middle finger force threshole setting value
						val4:Index finger force threshole setting value
						val5:Thumb force threshole setting value
						val6:Thumb swing force threshole setting value
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/

int Write_Hand_Force_Threshold(uint8_t ID, uint16_t val1, uint16_t val2, uint16_t val3, uint16_t val4, uint16_t val5, uint16_t val6)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x0F;
	tUartData.m_send_array[4] = kCmd_Handg3_Write;
	tUartData.m_send_array[5] = 0xDA;
	tUartData.m_send_array[6] = 0x05;
	tUartData.m_send_array[7] = val1 & 0xFF;
	tUartData.m_send_array[8] = val1 >> 8;
	tUartData.m_send_array[9] = val2 & 0xFF;
	tUartData.m_send_array[10] = val2 >> 8;
	tUartData.m_send_array[11] = val3 & 0xFF;
	tUartData.m_send_array[12] = val3 >> 8;
	tUartData.m_send_array[13] = val4 & 0xFF;
	tUartData.m_send_array[14] = val4 >> 8;
	tUartData.m_send_array[15] = val5 & 0xFF;
	tUartData.m_send_array[16] = val5 >> 8;
	tUartData.m_send_array[17] = val6 & 0xFF;
	tUartData.m_send_array[18] = val6 >> 8;
	for (i = 2; i < 19; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[19] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 20;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Write_Hand_Speed(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6)
** Descriptions:        Through register set the speed of each finger
** input parameters:    Input ID of the drive
						val1:Little figure speed setting value
						val2:Ring finger speed setting value
						val3:Middle finger speed setting value
						val4:Index finger speed setting value
						val5:Thumb speed setting value
						val6:Thumb swing speed setting value
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/

int Write_Hand_Speed(uint8_t ID, uint16_t val1, uint16_t val2, uint16_t val3, uint16_t val4, uint16_t val5, uint16_t val6)
{
	uint8_t i = 0;
	uint8_t m_unChecksum = 0;
	tUartData.m_send_array[0] = kSend_Frame_Head1;
	tUartData.m_send_array[1] = kSend_Frame_Head2;
	tUartData.m_send_array[2] = ID;
	tUartData.m_send_array[3] = 0x0F;
	tUartData.m_send_array[4] = kCmd_Handg3_Write;
	tUartData.m_send_array[5] = 0xF2;
	tUartData.m_send_array[6] = 0x05;
	tUartData.m_send_array[7] = val1 & 0xFF;
	tUartData.m_send_array[8] = val1 >> 8;
	tUartData.m_send_array[9] = val2 & 0xFF;
	tUartData.m_send_array[10] = val2 >> 8;
	tUartData.m_send_array[11] = val3 & 0xFF;
	tUartData.m_send_array[12] = val3 >> 8;
	tUartData.m_send_array[13] = val4 & 0xFF;
	tUartData.m_send_array[14] = val4 >> 8;
	tUartData.m_send_array[15] = val5 & 0xFF;
	tUartData.m_send_array[16] = val5 >> 8;
	tUartData.m_send_array[17] = val6 & 0xFF;
	tUartData.m_send_array[18] = val6 >> 8;
	for (i = 2; i < 19; i++)
	{
		m_unChecksum += tUartData.m_send_array[i];
	}
	tUartData.m_send_array[19] = m_unChecksum & 0xFF;
	tUartData.m_tx_len = 20;
	return 0;
}

/*********************************************************************************************************
** Function name:       void Uart_Rece_Data_Parsing()
** Descriptions:        Parse hand data
** input parameters:    None
** output parameters:   Hand data
** Returned value:      None
*********************************************************************************************************/

int Uart_Rece_Data_Parsing()
{
	uint8_t i = 0, j = 0;
	uint8_t m_unChecksum = 0;
	uint8_t m_unRxdata_usart = 0;
	uint8_t m_unRx_cnt_uart = 0;
	uint8_t m_unRx_data_uartcom = 0;
	uint16_t m_unUart_rx_data_lens = 0;
	for (j = 0; j < g_unNum; j++)
	{
		m_unRxdata_usart = tUartData.m_rec_array[j];
		m_unRx_data_uartcom = m_unRxdata_usart;
		if (m_unRx_cnt_uart == 0)
		{
			if (m_unRx_data_uartcom == kRcv_Frame_Head1)
			{
				tUartData.m_rec_array[m_unRx_cnt_uart] = m_unRx_data_uartcom;
				m_unRx_cnt_uart++;
			}
		}
		else if (m_unRx_cnt_uart == 1)
		{
			if (m_unRx_data_uartcom == kRcv_Frame_Head2)
			{
				tUartData.m_rec_array[m_unRx_cnt_uart] = m_unRx_data_uartcom;
				m_unRx_cnt_uart++;
			}
			else if (m_unRx_data_uartcom == kRcv_Frame_Head1)
			{
				m_unRx_cnt_uart = 0;
				tUartData.m_rec_array[m_unRx_cnt_uart] = m_unRx_data_uartcom;
				m_unRx_cnt_uart++;
			}
		}
		else if (m_unRx_cnt_uart == 2)
		{
			tUartData.m_rec_array[m_unRx_cnt_uart] = m_unRx_data_uartcom;
			m_unRx_cnt_uart++;
		}
		else if (m_unRx_cnt_uart == 3)
		{
			tUartData.m_rec_array[m_unRx_cnt_uart] = m_unRx_data_uartcom;
			m_unUart_rx_data_lens = m_unRx_data_uartcom;
			m_unRx_cnt_uart++;
		}
		else if (m_unRx_cnt_uart == (m_unUart_rx_data_lens + 4))
		{
			tUartData.m_rec_array[m_unRx_cnt_uart] = m_unRx_data_uartcom;
			m_unRx_cnt_uart = 0;
			for (i = 2; i < (m_unUart_rx_data_lens + 4); i++)
			{
				m_unChecksum += tUartData.m_rec_array[i];
			}
			if (m_unChecksum == tUartData.m_rec_array[m_unUart_rx_data_lens + 4])
			{
				g_unHand_id = tUartData.m_rec_array[2];
				if (tUartData.m_rec_array[4] == kCmd_Handg3_Read)
				{
					tHand.tLittleFinger.m_current_angle = (tUartData.m_rec_array[7] & 0xFF) + (tUartData.m_rec_array[8] << 8);
					tHand.tRingFinger.m_current_angle = (tUartData.m_rec_array[9] & 0xFF) + (tUartData.m_rec_array[10] << 8);
					tHand.tMiddleFinger.m_current_angle = (tUartData.m_rec_array[11] & 0xFF) + (tUartData.m_rec_array[12] << 8);
					tHand.tIndexFinger.m_current_angle = (tUartData.m_rec_array[13] & 0xFF) + (tUartData.m_rec_array[14] << 8);
					tHand.tThumbBend.m_current_angle = (tUartData.m_rec_array[15] & 0xFF) + (tUartData.m_rec_array[16] << 8);
					tHand.tThumbSide.m_current_angle = (tUartData.m_rec_array[17] & 0xFF) + (tUartData.m_rec_array[18] << 8);
					tHand.tLittleFinger.m_current_current = (tUartData.m_rec_array[55] & 0xFF) + (tUartData.m_rec_array[56] << 8);
					tHand.tRingFinger.m_current_current = (tUartData.m_rec_array[57] & 0xFF) + (tUartData.m_rec_array[58] << 8);
					tHand.tMiddleFinger.m_current_current = (tUartData.m_rec_array[59] & 0xFF) + (tUartData.m_rec_array[60] << 8);
					tHand.tIndexFinger.m_current_current = (tUartData.m_rec_array[61] & 0xFF) + (tUartData.m_rec_array[62] << 8);
					tHand.tThumbBend.m_current_current = (tUartData.m_rec_array[63] & 0xFF) + (tUartData.m_rec_array[64] << 8);
					tHand.tThumbSide.m_current_current = (tUartData.m_rec_array[65] & 0xFF) + (tUartData.m_rec_array[66] << 8);
					tHand.tLittleFinger.m_current_forceact = (tUartData.m_rec_array[43] & 0xFF) + (tUartData.m_rec_array[44] << 8);
					tHand.tRingFinger.m_current_forceact = (tUartData.m_rec_array[45] & 0xFF) + (tUartData.m_rec_array[46] << 8);
					tHand.tMiddleFinger.m_current_forceact = (tUartData.m_rec_array[47] & 0xFF) + (tUartData.m_rec_array[48] << 8);
					tHand.tIndexFinger.m_current_forceact = (tUartData.m_rec_array[49] & 0xFF) + (tUartData.m_rec_array[50] << 8);
					tHand.tThumbBend.m_current_forceact = (tUartData.m_rec_array[51] & 0xFF) + (tUartData.m_rec_array[52] << 8);
					tHand.tThumbSide.m_current_forceact = (tUartData.m_rec_array[53] & 0xFF) + (tUartData.m_rec_array[54] << 8);
					tHand.tLittleFinger.m_error_code = (tUartData.m_rec_array[67] & 0xFF);
					tHand.tRingFinger.m_error_code = (tUartData.m_rec_array[68] & 0xFF);
					tHand.tMiddleFinger.m_error_code = (tUartData.m_rec_array[69] & 0xFF);
					tHand.tIndexFinger.m_error_code = (tUartData.m_rec_array[70] & 0xFF);
					tHand.tThumbBend.m_error_code = (tUartData.m_rec_array[71] & 0xFF);
					tHand.tThumbSide.m_error_code = (tUartData.m_rec_array[72] & 0xFF);
				}
				else if (tUartData.m_rec_array[4] == kCmd_Mc_Angle_Force)
				{
					tHand.tLittleFinger.m_current_angle = (tUartData.m_rec_array[5] & 0xFF) + (tUartData.m_rec_array[6] << 8);
					tHand.tRingFinger.m_current_angle = (tUartData.m_rec_array[9] & 0xFF) + (tUartData.m_rec_array[10] << 8);
					tHand.tMiddleFinger.m_current_angle = (tUartData.m_rec_array[13] & 0xFF) + (tUartData.m_rec_array[14] << 8);
					tHand.tIndexFinger.m_current_angle = (tUartData.m_rec_array[17] & 0xFF) + (tUartData.m_rec_array[18] << 8);
					tHand.tThumbBend.m_current_angle = (tUartData.m_rec_array[21] & 0xFF) + (tUartData.m_rec_array[22] << 8);
					tHand.tThumbSide.m_current_angle = (tUartData.m_rec_array[25] & 0xFF) + (tUartData.m_rec_array[26] << 8);
					tHand.tLittleFinger.m_current_forceact = (tUartData.m_rec_array[7] & 0xFF) + (tUartData.m_rec_array[8] << 8);
					tHand.tRingFinger.m_current_forceact = (tUartData.m_rec_array[11] & 0xFF) + (tUartData.m_rec_array[12] << 8);
					tHand.tMiddleFinger.m_current_forceact = (tUartData.m_rec_array[15] & 0xFF) + (tUartData.m_rec_array[16] << 8);
					tHand.tIndexFinger.m_current_forceact = (tUartData.m_rec_array[19] & 0xFF) + (tUartData.m_rec_array[20] << 8);
					tHand.tThumbBend.m_current_forceact = (tUartData.m_rec_array[23] & 0xFF) + (tUartData.m_rec_array[24] << 8);
					tHand.tThumbSide.m_current_forceact = (tUartData.m_rec_array[27] & 0xFF) + (tUartData.m_rec_array[28] << 8);
				}
				else if (tUartData.m_rec_array[4] == kCmd_Mc_Force)
				{
					tHand.tLittleFinger.m_current_forceact = (tUartData.m_rec_array[7] & 0xFF) + (tUartData.m_rec_array[8] << 8);
					tHand.tRingFinger.m_current_forceact = (tUartData.m_rec_array[9] & 0xFF) + (tUartData.m_rec_array[10] << 8);
					tHand.tMiddleFinger.m_current_forceact = (tUartData.m_rec_array[11] & 0xFF) + (tUartData.m_rec_array[12] << 8);
					tHand.tIndexFinger.m_current_forceact = (tUartData.m_rec_array[13] & 0xFF) + (tUartData.m_rec_array[14] << 8);
					tHand.tThumbBend.m_current_forceact = (tUartData.m_rec_array[15] & 0xFF) + (tUartData.m_rec_array[16] << 8);
					tHand.tThumbSide.m_current_forceact = (tUartData.m_rec_array[17] & 0xFF) + (tUartData.m_rec_array[18] << 8);
				}
				else if (tUartData.m_rec_array[4] == kCmd_Mc_Current)
				{
					tHand.tLittleFinger.m_current_current = (tUartData.m_rec_array[7] & 0xFF) + (tUartData.m_rec_array[8] << 8);
					tHand.tRingFinger.m_current_current = (tUartData.m_rec_array[9] & 0xFF) + (tUartData.m_rec_array[10] << 8);
					tHand.tMiddleFinger.m_current_current = (tUartData.m_rec_array[11] & 0xFF) + (tUartData.m_rec_array[12] << 8);
					tHand.tIndexFinger.m_current_current = (tUartData.m_rec_array[13] & 0xFF) + (tUartData.m_rec_array[14] << 8);
					tHand.tThumbBend.m_current_current = (tUartData.m_rec_array[15] & 0xFF) + (tUartData.m_rec_array[16] << 8);
					tHand.tThumbSide.m_current_current = (tUartData.m_rec_array[17] & 0xFF) + (tUartData.m_rec_array[18] << 8);
				}
				else if (tUartData.m_rec_array[4] == kCmd_Mc_All)
				{
					tHand.tLittleFinger.m_current_angle = (tUartData.m_rec_array[5] & 0xFF) + (tUartData.m_rec_array[6] << 8);
					tHand.tRingFinger.m_current_angle = (tUartData.m_rec_array[12] & 0xFF) + (tUartData.m_rec_array[13] << 8);
					tHand.tMiddleFinger.m_current_angle = (tUartData.m_rec_array[19] & 0xFF) + (tUartData.m_rec_array[20] << 8);
					tHand.tIndexFinger.m_current_angle = (tUartData.m_rec_array[26] & 0xFF) + (tUartData.m_rec_array[27] << 8);
					tHand.tThumbBend.m_current_angle = (tUartData.m_rec_array[33] & 0xFF) + (tUartData.m_rec_array[34] << 8);
					tHand.tThumbSide.m_current_angle = (tUartData.m_rec_array[40] & 0xFF) + (tUartData.m_rec_array[41] << 8);
					tHand.tLittleFinger.m_current_current = (tUartData.m_rec_array[7] & 0xFF) + (tUartData.m_rec_array[8] << 8);
					tHand.tRingFinger.m_current_current = (tUartData.m_rec_array[14] & 0xFF) + (tUartData.m_rec_array[15] << 8);
					tHand.tMiddleFinger.m_current_current = (tUartData.m_rec_array[21] & 0xFF) + (tUartData.m_rec_array[22] << 8);
					tHand.tIndexFinger.m_current_current = (tUartData.m_rec_array[28] & 0xFF) + (tUartData.m_rec_array[29] << 8);
					tHand.tThumbBend.m_current_current = (tUartData.m_rec_array[35] & 0xFF) + (tUartData.m_rec_array[36] << 8);
					tHand.tThumbSide.m_current_current = (tUartData.m_rec_array[42] & 0xFF) + (tUartData.m_rec_array[43] << 8);
					tHand.tLittleFinger.m_current_forceact = (tUartData.m_rec_array[9] & 0xFF) + (tUartData.m_rec_array[10] << 8);
					tHand.tRingFinger.m_current_forceact = (tUartData.m_rec_array[16] & 0xFF) + (tUartData.m_rec_array[17] << 8);
					tHand.tMiddleFinger.m_current_forceact = (tUartData.m_rec_array[23] & 0xFF) + (tUartData.m_rec_array[24] << 8);
					tHand.tIndexFinger.m_current_forceact = (tUartData.m_rec_array[30] & 0xFF) + (tUartData.m_rec_array[31] << 8);
					tHand.tThumbBend.m_current_forceact = (tUartData.m_rec_array[37] & 0xFF) + (tUartData.m_rec_array[38] << 8);
					tHand.tThumbSide.m_current_forceact = (tUartData.m_rec_array[44] & 0xFF) + (tUartData.m_rec_array[45] << 8);
					tHand.tLittleFinger.m_error_code = (tUartData.m_rec_array[11] & 0xFF);
					tHand.tRingFinger.m_error_code = (tUartData.m_rec_array[18] & 0xFF);
					tHand.tMiddleFinger.m_error_code = (tUartData.m_rec_array[25] & 0xFF);
					tHand.tIndexFinger.m_error_code = (tUartData.m_rec_array[32] & 0xFF);
					tHand.tThumbBend.m_error_code = (tUartData.m_rec_array[39] & 0xFF);
					tHand.tThumbSide.m_error_code = (tUartData.m_rec_array[46] & 0xFF);
				}
			}
		}
		else
		{
			tUartData.m_rec_array[m_unRx_cnt_uart] = m_unRx_data_uartcom;
			m_unRx_cnt_uart++;
		}
	}
}
