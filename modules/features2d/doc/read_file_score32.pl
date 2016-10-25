#!/usr/bin/perl
use strict;
use warnings;
use autodie; # die if problem reading or writing a file

my $filein = "./agast_score.txt";
my $fileout = "./agast_new.txt";
my $i1=1;
my $i2=1;
my $i3=1;
my $tmp;
my $ifcount0=0;
my $ifcount1=0;
my $ifcount2=0;
my $ifcount3=0;
my $ifcount4=0;
my $elsecount;
my $myfirstline = $ARGV[0];
my $mylastline  = $ARGV[1];
my $tablename  = $ARGV[2];
my @array0 = ();
my @array1 = ();
my @array2 = ();
my @array3 = ();
my $is_not_a_corner;
my $is_a_corner;

 open(my $in1,  "<",  $filein)  or die "Can't open $filein: $!";
 open(my $out, ">",  $fileout) or die "Can't open $fileout: $!";


    $array0[0] = 0;
    $i1=1;
    while (my $line1 = <$in1>)
    {
        chomp $line1;
    $array0[$i1] = 0;
    if (($i1>=$myfirstline)&&($i1<=$mylastline))
    {
                if($line1=~/if\(ptr\[offset(\d+)/)
                {
                  if($line1=~/if\(ptr\[offset(\d+).*\>.*cb/)
          {
            $tmp=$1;
                  }
          else
                  {
                   if($line1=~/if\(ptr\[offset(\d+).*\<.*c\_b/)
           {
            $tmp=$1+128;
                   }
                   else
           {
                        die "invalid array index!"
           }
          }
          $array1[$ifcount1] = $tmp;
          $array0[$ifcount1] = $i1;
          $ifcount1++;
        }
        else
        {
        }
    }
        $i1++;
    }
   $is_not_a_corner=$ifcount1;
   $is_a_corner=$ifcount1+1;

  close $in1 or die "Can't close $filein: $!";

  open($in1,  "<",  $filein)  or die "Can't open $filein: $!";


    $i1=1;
    while (my $line1 = <$in1>)
    {
        chomp $line1;
    if (($i1>=$myfirstline)&&($i1<=$mylastline))
    {
           if ($array0[$ifcount2] == $i1)
       {
        $array2[$ifcount2]=0;
        $array3[$ifcount2]=0;
                if ($array0[$ifcount2+1] == ($i1+1))
            {
          $array2[$ifcount2]=($ifcount2+1);
            }
        else
            {
                  open(my $in2,  "<",  $filein)  or die "Can't open $filein: $!";
          $i2=1;
                  while (my $line2 = <$in2>)
                  {
                      chomp $line2;
                      if ($i2 == $i1)
                      {
                          last;
                      }
                      $i2++;
                  }
                  my $line2 = <$in2>;
                  chomp $line2;
                  if ($line2=~/goto (\w+)/)
                  {
                               $tmp=$1;
                               if ($tmp eq "is_not_a_corner")
                               {
                                   $array2[$ifcount2]=$is_not_a_corner;
                               }
                               if ($tmp eq "is_a_corner")
                               {
                                   $array2[$ifcount2]=$is_a_corner;
                               }
                  }
                  else
                  {
                      die "goto expected: $!";
                  }
                  close $in2 or die "Can't close $filein: $!";
                }
                #find next else and interprete it
                open(my $in3,  "<",  $filein)  or die "Can't open $filein: $!";
        $i3=1;
        $ifcount3=0;
                $elsecount=0;
                while (my $line3 = <$in3>)
                {
                      chomp $line3;
                      $i3++;
                      if ($i3 == $i1)
                      {
                          last;
                      }
                }
                while (my $line3 = <$in3>)
                {
                      chomp $line3;
                      $ifcount3++;
                      if (($elsecount==0)&&($i3>$i1))
                      {
                            if ($line3=~/goto (\w+)/)
                            {
                               $tmp=$1;
                               if ($tmp eq "is_not_a_corner")
                               {
                                   $array3[$ifcount2]=$is_not_a_corner;
                               }
                               if ($tmp eq "is_a_corner")
                               {
                                   $array3[$ifcount2]=$is_a_corner;
                               }
                            }
                            else
                            {
                                if ($line3=~/if\(ptr\[offset/)
                                {
                                        $ifcount4=0;
                                        while ($array0[$ifcount4]!=$i3)
                                        {
                                            $ifcount4++;
                                            if ($ifcount4==$ifcount1)
                                            {
                                          die "if else match expected: $!";
                                            }
                                            $array3[$ifcount2]=$ifcount4;
                                        }
                                }
                                else
                                {
                                        die "elseif or elsegoto match expected: $!";
                                }
                            }
                            last;
                      }
                      else
                      {
                            if ($line3=~/if\(ptr\[offset/)
                            {
                              $elsecount++;
                            }
                            else
                            {
                                if ($line3=~/else/)
                                {
                                  $elsecount--;
                                }
                            }
                      }
                      $i3++;
                }
                printf("%3d [%3d][0x%08x]\n", $array0[$ifcount2], $ifcount2, (($array1[$ifcount2]&15)<<28)|($array2[$ifcount2]<<16)|(($array1[$ifcount2]&128)<<5)|($array3[$ifcount2]));
                close $in3 or die "Can't close $filein: $!";
        $ifcount2++;
       }
       else
       {
       }
    }
        $i1++;
    }

  printf("    [%3d][0x%08x]\n", $is_not_a_corner, 254);
  printf("    [%3d][0x%08x]\n", $is_a_corner, 255);

  close $in1 or die "Can't close $filein: $!";

  $ifcount0=0;
  $ifcount2=0;
  printf $out "    static const unsigned long %s[] = {\n        ", $tablename;
  while ($ifcount0 < $ifcount1)
  {
      printf $out "0x%08x, ", (($array1[$ifcount0]&15)<<28)|($array2[$ifcount0]<<16)|(($array1[$ifcount0]&128)<<5)|($array3[$ifcount0]);

     $ifcount0++;
     $ifcount2++;
     if ($ifcount2==8)
     {
      $ifcount2=0;
      printf $out "\n";
      printf $out "        ";
     }

  }
  printf $out "0x%08x, ", 254;
     $ifcount0++;
     $ifcount2++;
     if ($ifcount2==8)
     {
      $ifcount2=0;
      printf $out "\n";
      printf $out "        ";
     }
  printf $out "0x%08x\n", 255;
     $ifcount0++;
     $ifcount2++;
      printf $out "    };\n\n";

  $#array0 = -1;
  $#array1 = -1;
  $#array2 = -1;
  $#array3 = -1;

 close $out or die "Can't close $fileout: $!";
