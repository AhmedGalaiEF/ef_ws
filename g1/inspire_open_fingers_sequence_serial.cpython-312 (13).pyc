#include <iostream>
#include <pthread.h>
#include <chrono> // Provides std::chrono
#include "hand_api.h"

int main()
{
    pthread_t tid;
    int ret = 0;
    unsigned int array = 0;
    char s;
    if ((fd_right = open("/dev/ttyUSB0", O_RDWR | O_NOCTTY)) < 0 || (fd_left = open("/dev/ttyUSB0", O_RDWR | O_NOCTTY)) < 0)
    {
        printf("Open failed\n");
    }
    else
    {
        set_opt(fd_right, 115200, 8, 'N', 1);
        set_opt(fd_left, 115200, 8, 'N', 1);
        ret = pthread_create(&tid, NULL, ThreadEntry_right, NULL);
        ret = pthread_create(&tid, NULL, ThreadEntry_left, NULL);
        if (ret != 0)
        {
            perror("pthread_create");
            exit(1);
        }
        while (1)
        {
            Action(0);

            //     // fgets(m_send_array, sizeof(m_send_array), stdin); // stdin means keyboard input
            //     // write(fd_right,tUartData.m_send_array,sizeof(*tUartData.m_send_array));
            //     // read(fd_right,tUartData.m_rec_array,sizeof(*tUartData.m_rec_array));*/
            // }
        }
    }
    close(fd_right);
    // fclose(fp);
    // fp = NULL; // Reset the pointer so it no longer points to the old file
    return 0;
}
