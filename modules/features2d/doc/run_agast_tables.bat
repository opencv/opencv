perl read_file_score32.pl 9059 9385 table_5_8_corner_struct
move agast_new.txt agast_score_table.txt
perl read_file_score32.pl 2215 3387 table_7_12d_corner_struct
copy /A agast_score_table.txt + agast_new.txt agast_score_table.txt
del agast_new.txt
perl read_file_score32.pl 3428 9022 table_7_12s_corner_struct
copy /A agast_score_table.txt + agast_new.txt agast_score_table.txt
del agast_new.txt
perl read_file_score32.pl 118 2174 table_9_16_corner_struct
copy /A agast_score_table.txt + agast_new.txt agast_score_table.txt
del agast_new.txt

perl read_file_nondiff32.pl 103 430 table_5_8_struct1
move agast_new.txt agast_table.txt
perl read_file_nondiff32.pl 440 779 table_5_8_struct2
copy /A agast_table.txt + agast_new.txt agast_table.txt
del agast_new.txt
perl read_file_nondiff32.pl 869 2042 table_7_12d_struct1
copy /A agast_table.txt + agast_new.txt agast_table.txt
del agast_new.txt
perl read_file_nondiff32.pl 2052 3225 table_7_12d_struct2
copy /A agast_table.txt + agast_new.txt agast_table.txt
del agast_new.txt
perl read_file_nondiff32.pl 3315 4344 table_7_12s_struct1
copy /A agast_table.txt + agast_new.txt agast_table.txt
del agast_new.txt
perl read_file_nondiff32.pl 4354 5308 table_7_12s_struct2
copy /A agast_table.txt + agast_new.txt agast_table.txt
del agast_new.txt
perl read_file_nondiff32.pl 5400 7454 table_9_16_struct
copy /A agast_table.txt + agast_new.txt agast_table.txt
del agast_new.txt
