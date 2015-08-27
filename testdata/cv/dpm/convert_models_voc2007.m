dir_mat = 'VOC2007/';
dir_xml = 'VOC2007_Cascade/';
mkdir(dir_xml);
fs = dir('VOC2007/*.mat');
for i = 1 : length(fs)
    fname = fs(i).name;
    if strcmp('person_grammar_final.mat', fname)
        continue;
    end
    fprintf('\n%s', fname);
    fname_in = [dir_mat fname];
    fname_out = [dir_xml fname(1:end-10) '.xml'];
    mat2opencvxml(fname_in, fname_out);
end