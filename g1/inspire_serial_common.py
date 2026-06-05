#ifndef _API_H_ 
#define _API_H_ 

#ifdef __cplusplus  
extern "C" {  
#endif

#include <stdint.h>
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
#include <pthread.h> 

// #endif

#define kSend_Frame_Head1				0xEB
#define kSend_Frame_Head2				0x90


#define kRcv_Frame_Head1				0x90
#define kRcv_Frame_Head2				0xEB


//Direct control flag

#define kCmd_Handg3_Read  			       0x11  //read register
#define kCmd_Handg3_Write  			       0x12	 //write register
#define kCmd_Mc_Angle_Force                          0x14  //motor motion enable
#define kCmd_Mc_Force                                0x15  //motor motion disable
#define kCmd_Mc_Current                              0x16  //save parameters to flash
#define kCmd_Mc_All                                  0x17  //get drive status


//Number of data                          
#define kUart_Recv_Length                            256   //data length



struct Figure
{
	  int16_t       m_current_angle;                   // Current finger angle
	  int16_t       m_current_current;                 // Current finger motor current
	  int16_t       m_current_forceact;                // Actual finger force
	  int8_t        m_error_code;                      // Finger error code
};

typedef struct 
{
	 struct Figure  tLittleFinger;                                  
	 struct Figure  tRingFinger;                      
	 struct Figure  tMiddleFinger;                     
	 struct Figure  tIndexFinger;                      
	 struct Figure  tThumbBend;                       
	 struct Figure  tThumbSide;                        
}ActuatorStatusTypeDef;

typedef struct _Uart_Data_Buff{
	uint8_t m_tx_len;
	uint8_t m_rec_array[128];     //Define receive data array
	uint8_t m_send_array[128];     //Define send data array
}UartDataBuff;



extern ActuatorStatusTypeDef tHand;
extern UartDataBuff tUartData;

extern volatile int fd_left,fd_right,nByte;
extern volatile uint8_t   sum,g_unHand_id;                   //Hand ID
extern volatile uint16_t  g_unNum;                      //Define data length

/*********************************************************************************************************
** Function name:       void*Action(int array)    
** Descriptions:        Action function     
** input parameters:    number of action
** output parameters:   none
** Returned value:      None
*********************************************************************************************************/
void*Action(int numOfAction);
/*********************************************************************************************************
** Function name:       void*ThreadEntry(void*arg)     
** Descriptions:        Thread function     
** input parameters:    None
** output parameters:   none
** Returned value:      None
*********************************************************************************************************/
void*ThreadEntry_left(void*arg);

void*ThreadEntry_right(void*arg);
/*********************************************************************************************************
** Function name:       void setTimer(int seconds, int mseconds, int com)     
** Descriptions:        Timing function     
** input parameters:    seconds: the seconds; mseconds: the micro seconds,com:the com
** output parameters:   none
** Returned value:      None
*********************************************************************************************************/
void setTimer(int seconds, int mseconds, int com);
/*********************************************************************************************************
** Function name:       int set_opt(int ,int , int , char , int )     
** Descriptions:        Serial port configuration function     
** input parameters:    com,speed,bit,even,stop
** output parameters:   none
** Returned value:      None
*********************************************************************************************************/
int set_opt(int ,int , int , char , int );
/*********************************************************************************************************
** Function name:       int Read_Hand_Data(uint8_t ID, uint8_t adress, uint8_t len)     
** Descriptions:        Read hand data      
** input parameters:    Input ID of the hand
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/
int Read_Hand_Data(uint8_t ID, uint16_t adress, uint16_t len);
/*********************************************************************************************************
** Function name:       int Save_Hand_Data(uint8_t ID)      
** Descriptions:        Save hand data      
** input parameters:    Input ID of the hand
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/
int Save_Hand_Data(u_int8_t ID);
/*********************************************************************************************************
** Function name:       int Read_Hand_Register_Data(uint8_t ID,uint16_t val,uint16_t vallen)      
** Descriptions:        Read hand register data      
** input parameters:    ID:Input ID of the drive
                        val:Input starting register value
                        vallen:Input read register length
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/
int Read_Hand_Register_Data(uint8_t ID,uint16_t val,uint16_t vallen);
/*********************************************************************************************************
** Function name:       int Change_Hand_ID(uint8_t ID,uint8_t val)      
** Descriptions:        Change hand ID      
** input parameters:    ID:Input ID of the drive
                        val:Input another ID
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/
int Change_Hand_ID(uint8_t ID,uint8_t val);
/*********************************************************************************************************
** Function name:       int Change_Hand_Baud(uint8_t ID,uint8_t baud)      
** Descriptions:        Change baud of the hand      
** input parameters:    ID:ID of the current hand
                        baud:Representative number of baud rate,0--19200 1--57600 2--115200 
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/
int Change_Hand_Baud(uint8_t ID,uint8_t baud);
/*********************************************************************************************************
** Function name:       int Clear_Hand_Error(uint8_t ID)      
** Descriptions:        Clear drive error      
** input parameters:    Input ID of the drive
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/
int Clear_Hand_Error(uint8_t ID);
/*********************************************************************************************************
** Function name:       int Reset_Hand(uint8_t ID)      
** Descriptions:        Reset drive       
** input parameters:    Input ID of the drive
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/
int Reset_Hand(uint8_t ID);
/*********************************************************************************************************
** Function name:       int Force_Sensor_Calibration(uint8_t ID)      
** Descriptions:        Calibrate the force sensor      
** input parameters:    Input ID of the drive
** output parameters:   Outout command array
** Returned value:      None
*********************************************************************************************************/
int Force_Sensor_Calibration(uint8_t ID);
/*********************************************************************************************************
** Function name:       int Write_Hand_Angle(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6)      
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
int Write_Hand_Angle(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6);
/*********************************************************************************************************
** Function name:       int Write_Hand_Drive_Position(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6)      
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
int Write_Hand_Drive_Position(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6);
/*********************************************************************************************************
** Function name:       int Write_Hand_Angle_Position(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6)      
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
int Write_Hand_Angle_Position(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6);
/*********************************************************************************************************
** Function name:       int Write_Hand_Force_Threshold(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6)      
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
int Write_Hand_Force_Threshold(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6);
/*********************************************************************************************************
** Function name:       int Write_Hand_Speed(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6)      
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
int Write_Hand_Speed(uint8_t ID,uint16_t val1,uint16_t val2,uint16_t val3,uint16_t val4,uint16_t val5,uint16_t val6);
/*********************************************************************************************************
** Function name:       int Uart_Rece_Data_Parsing()
** Descriptions:        Parse hand data
** input parameters:    None
** output parameters:   Hand data
** Returned value:      None
*********************************************************************************************************/
int Uart_Rece_Data_Parsing();


#ifdef __cplusplus  
}  
#endif  

#endif // _API_H_ 
