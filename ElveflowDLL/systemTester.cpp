#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <csignal>  // Correctly include csignal for signal handling
#include <time.h>
#include <windows.h>
#include "error_check.h"
#include "OB1.h"

// Include Elveflow header
#include "dll/Elveflow64.h"

using namespace std;

#define MAX_LINE_LENGTH 100

int pressure = 0;
int cur_time = 0; // Missing semicolon
int MyOB1_ID = -3;
int channel = 1;
double *Calibration = new double[1000];
int MyM_S_R_D_ID = -1;

// Dummy function to process data
int get_pressure() {
    double* pres = new double[4];
    int a;
    if ((a = OB1_Get_Press(MyOB1_ID, channel, 1, Calibration, pres, 1000)) != 0) {
        fprintf(stderr, "error getting pressure %d", a);
    } //use pointer
    return *pres;
}

void set_pressure(int a) {
    int b;
    if ((b = OB1_Set_Press(MyOB1_ID, channel, a, Calibration, 1000)) != 0) {
        fprintf(stderr, "error setting pressure %d", b);
    }
}

int get_flow() {
    double get_Sens_data;
    int a;
    if ((a = M_S_R_D_Get_Sens_Data(MyM_S_R_D_ID, channel, &get_Sens_data)) != 0) {
        fprintf(stderr, "error reading flow sensor %d", a);
    }
    return get_Sens_data;
}

void handle_signal(int signum) {
    printf("\nReceived signal: %d\n", signum);
    if (signum == SIGINT) {
        printf("Interrupt signal received, performing cleanup...\n");
        OB1_Destructor(MyOB1_ID);
        M_S_R_D_Destructor(MyM_S_R_D_ID);
        system("PAUSE");
        printf("CLEANUP DONE \n");
        // Perform cleanup here
        exit(0);  // Exit the program gracefully
    }
}

FILE* initialize() {
    char name[] = "02062435";
    int error = OB1_Initialization(name, 2, 0, 0, 0, &MyOB1_ID);
    Check_Error(error);

    if (MyOB1_ID < 0) {
        perror("Error with MY OB1 ID \n");
        printf("MY OB1 ID: %d", MyOB1_ID);
        exit(EXIT_FAILURE);
    }

    Elveflow_Calibration_Load("Calib_08-14", Calibration, 1000);

    // cout << "wait ~2 minutes" << endl;
    // OB1_Calib(MyOB1_ID, Calibration, 1000);//Perform calibration ! ! ! Take about 2 minutes ! ! !
    // printf("calib done \n");
    // error = Elveflow_Calibration_Save("Calib_08-14", Calibration, 1000); //save calibration in the selected path, if no path or non valid path, open prompt to ask the file name
    // Check_Error(error);
    // printf("save done \n");
    // OB1_Destructor(MyOB1_ID);
    // M_S_R_D_Destructor(MyM_S_R_D_ID);
    // system("PAUSE");
    // exit(EXIT_SUCCESS);

    int msrd_error = M_S_R_D_Initialization("02062435", 2, 0, 0, 0, 0, 0, &MyM_S_R_D_ID);
    Check_Error(msrd_error);
    if (MyM_S_R_D_ID < 0) {
        perror("Error with MY MSRD ID \n");
        printf("MY MSRD ID: %d", MyM_S_R_D_ID);
        exit(EXIT_FAILURE);
    }

    error = M_S_R_D_Add_Sens(MyM_S_R_D_ID, 1, 5, 1, 1, 7);
    Check_Error(error);
    // Open a file in read mode
    FILE* data = fopen("slanting_test_data.txt", "r");

    if (data == NULL) {
        perror("Error opening file DATA NULL?");
        exit(EXIT_FAILURE);
    }

    if (signal(SIGINT, handle_signal) == SIG_ERR) {
        perror("Error setting up signal handler");
    }
    return data;
}
int main() {
    
    FILE* data = initialize();

    char line[MAX_LINE_LENGTH];
    clock_t test_time = clock();
    while (clock() < test_time + 30 * CLOCKS_PER_SEC) {
        set_pressure(10);
    }
    printf("done with initial test run at 10");
    while (clock() < test_time + 30 * CLOCKS_PER_SEC) {
        set_pressure(0);
    }
    printf("done with 0 run, time to start!");
    double reiman_sum = 0;
    clock_t very_start_time = clock();
    FILE* output = fopen("slanting_output_08-13.txt", "w");
    if (output == NULL) {
        perror("Error opening file DATA NULL?");
        exit(EXIT_FAILURE);
    }
    while (fgets(line, sizeof(line), data)) { // Read each line

        // Use double for pressure and int for time
        double pressure_input;
        double t_input;

        // Use %lf for double and %d for int
        if (sscanf(line, "%lf %lf", &pressure_input, &t_input) != 2) { // Parse the line
            fprintf(stderr, "Error reading line: %s", line);
            continue;
        }
        clock_t start_time = clock();
        clock_t total_time = 0;

        clock_t prev_time = start_time;
        

        while (total_time < t_input * CLOCKS_PER_SEC) {
            double cur_time = clock();
            if ((cur_time - prev_time) >= 0.01 * CLOCKS_PER_SEC) {
                set_pressure(static_cast<int>(pressure_input)); // Cast pressure_input to int
                double pres = get_pressure();
                double flow = get_flow();
                reiman_sum += flow*(cur_time - prev_time) / CLOCKS_PER_SEC;
                //time interval, pressure input, actual pressure, flow, reimann sum
                fprintf(output, "%f\t%d\t%f\t%f\t%f\n", (cur_time - very_start_time) / CLOCKS_PER_SEC, static_cast<int>(pressure_input), pres, flow, reiman_sum);

                prev_time = cur_time;
            }
            total_time = cur_time - start_time;
        }
    

        // Process data using dummy function
    }
    fclose(output);
    fclose(data); // Close the file
    free(Calibration);
    OB1_Destructor(MyOB1_ID);
    M_S_R_D_Destructor(MyM_S_R_D_ID);
    system("PAUSE");
    return 0;
}
