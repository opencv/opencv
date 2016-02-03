/*
* Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
* Copyright (c) 2002-2007, Professor Benoit Macq
* Copyright (c) 2003-2007, Francois-Olivier Devaux 
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */
#include <stdio.h>
#include <string.h>
#include "md5.h"

#define OPJ_Bin_Dir "OPJ_Binaries"

int doprocess(char programme[4096],char command_line[4096]) {

#ifdef _WIN32
	
	int exit=STILL_ACTIVE;
	STARTUPINFO siStartupInfo;
	PROCESS_INFORMATION piProcessInfo;
	
	memset(&siStartupInfo, 0, sizeof(siStartupInfo));
	memset(&piProcessInfo, 0, sizeof(piProcessInfo));
	siStartupInfo.cb = sizeof(siStartupInfo);
	
	if(CreateProcess(programme, // Application name
		command_line, // Application arguments
		0,
		0,
		FALSE,
		CREATE_DEFAULT_ERROR_MODE,
		0,
		0, // Working directory
		&siStartupInfo,
		&piProcessInfo) == FALSE)	
		return 1;
	
	exit=STILL_ACTIVE;
	while(exit==STILL_ACTIVE) {
		Sleep(1000);
		GetExitCodeProcess(piProcessInfo.hProcess,&exit);
	}
	
	return 0;

#else /* !_WIN32 */
	printf("\n%s\n", command_line);
	system(command_line);
	return 0;

#endif /* _WIN32 */
	
}

char MD5_process(char *input_filename, char *md5_filename) {
	MD5_CTX mdContext;
	int bytes;
  unsigned char data[1024];
	FILE *input_file, *md5_file;
	
	input_file = fopen(input_filename, "rb");
	if (!input_file) {
		printf("Error opening file %s\n", input_filename);
		return 1;
	}
	
	md5_file = fopen(md5_filename, "wb");
	if (!md5_file) {
		printf("Error opening file %s\n", md5_filename);
		return 1;
	}
	
	MD5Init (&mdContext);
  while ((bytes = fread (data, 1, 1024, input_file)) != 0)
    MD5Update (&mdContext, data, bytes);
  MD5Final (&mdContext);
	
	fwrite(mdContext.digest,16,1,md5_file);
	
	fclose(input_file);
	fclose(md5_file);
	
	return 0;
}

char fcompare(char *input_filename, char *output_filename) {
	FILE *input_file, *output_file;
	unsigned char input_buffer[17], output_buffer[17];
	char comparison;
	
	input_file = fopen(input_filename, "rb");
	if (!input_file) {
		printf("Error opening file %s\n", input_filename);
		return -1;
	}
	
	output_file = fopen(output_filename, "rb");
	if (!output_file) {
		printf("Error opening file %s\n", output_filename);
		return -1;
	}
	
	fread(input_buffer,16,1,input_file);
	fread(output_buffer,16,1,output_file);
	fclose(input_file);
	fclose(output_file);
	input_buffer[16] = 0;
	output_buffer[16] = 0;
	
	comparison = strcmp(input_buffer, output_buffer);
	
	if (comparison)
		return 1;
	return 0;
}

int main(int argc, char* argv[]) {
	FILE *param_file, *md5_file;
	FILE *report_file;
	char line[4096];
	char md5_filename[4096], tempmd5_filename[4096], temp[4096], report_filename[4096];
	char output_filename[4096];
	char input_cmdline[4096];
	char command_line[4096], exefile[4096];
	int task_counter = 0, word_counter;
	char bin_dir[4096];
	unsigned int word_pointer;
	char ch[4096];				
	char comparison;
	int num_failed = 0;
	int num_inexistant = 0;
	int num_passed = 0;
		
	if (argc != 3) {
		printf("Error with command line. \nExpected: OPJ_Validate parameter_filename bin_directory\n Example: OPJ_Validate parameters_01.txt version1.1.a\n\n");
		return 1;
	}
	
	param_file = fopen(argv[1],"rb");
	if (!param_file) {
		printf("Error opening parameter file %s\n",argv[1]);
		return 1;
	}	
	
	sprintf(bin_dir,"%s/%s",OPJ_Bin_Dir,argv[2]);
	sprintf(tempmd5_filename,"temp/tempmd5.txt");
	sprintf(report_filename,"%s/report.txt",bin_dir);
	report_file = fopen(report_filename, "wb");
	if (!report_file) {
		printf("Unable to open report file %s", report_filename);
		return 1;
	}
	
	while (fgets(line, 4096, param_file) != NULL) {
		
		if (line[0] != '#' && line[0] != 0x0d) {	// If not a comment line
			sscanf(line, "%s", temp);
			word_pointer = 0;
			sprintf(input_cmdline,"");	
			sscanf(line+word_pointer,"%s",ch);
			sprintf(exefile,"%s/%s",bin_dir,ch);				
			word_counter = 0;
			while (sscanf(line+word_pointer,"%s",ch) > 0) {
				if (word_counter == 4) 
					strcpy(output_filename, ch);
				word_pointer += strlen(ch)+1;
				sprintf(input_cmdline,"%s%s ",input_cmdline, ch);				
				word_counter++;
			}			
			sprintf(md5_filename,"%s.md5",output_filename);
			task_counter++;
			sprintf(command_line,"%s/%s",bin_dir,input_cmdline);
			printf("Task %d\nMD5 file: %s\nCommand line: \"%s\"\n",task_counter, md5_filename,command_line);
			fprintf(report_file,"Task %d\n   MD5 file: %s\n   Command line: \"%s\"\n",task_counter, md5_filename,command_line);
			
			if (doprocess(exefile,command_line)) {
				printf("Error executing: \"%s\" \n", command_line);
				fprintf(report_file,"Task %d failed because command line is not valid.\n\n", task_counter);
			}
			else {
				
				// Check if MD5 reference exists
				md5_file = fopen(md5_filename,"rb");
				if (md5_file) {
					fclose(md5_file);
					if (MD5_process(output_filename, tempmd5_filename)) 
						return 1;
					
					comparison = fcompare(tempmd5_filename, md5_filename);
					if (comparison == -1)
						return 1;
					else if (comparison) {
						printf("ERROR: %s and %s are different.\nThe codec seems to behave differently.\n\n", tempmd5_filename, md5_filename);
						fprintf(report_file,"ERROR: %s and %s are different.\nThe codec seems to behave differently.\n\n", tempmd5_filename, md5_filename);
						num_failed++;
					}
					else {
						printf("%s and %s are the same.\nTask %d OK\n\n",tempmd5_filename, md5_filename, task_counter);
						fprintf(report_file,"   %s and %s are the same.\nTask %d OK\n\n",tempmd5_filename, md5_filename, task_counter);
						num_passed++;
					}
					remove(tempmd5_filename);
				}	
				else {
					if (MD5_process(output_filename, md5_filename))
						return 1;
					printf("...  MD5 of %s was inexistant. It has been created\n\n", output_filename);
					fprintf(report_file,"MD5 of %s was inexistant. It has been created\n\n", output_filename);
					num_inexistant++;
				}
			}
		}
	}		

	printf("\nREPORT;\n%d tests num_passed\n%d tests num_failed\n%d MD5 were num_inexistant\n", num_passed, num_failed, num_inexistant);
	fprintf(report_file,"\nREPORT;\n%d tests num_passed\n%d tests num_failed\n%d MD5 were num_inexistant\n", num_passed, num_failed, num_inexistant);
	fclose(param_file);
	fclose(report_file);
		
}
