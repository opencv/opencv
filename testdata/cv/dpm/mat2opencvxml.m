function mat2opencvxml(fname_in, fname_out)
% load VOC2007 DPM model

load(fname_in);

thresh = -0.5;
pca = 5;
csc_model = cascade_model(model, '2007', pca, thresh);

num_feat = 32;
rootfilters = [];
for i = 1:length(csc_model.rootfilters)
    rootfilters{i} = csc_model.rootfilters{i}.w;
end
partfilters = [];
for i = 1:length(csc_model.partfilters)
    partfilters{i} = csc_model.partfilters{i}.w;
end
for c = 1:csc_model.numcomponents
    ridx{c} = csc_model.components{c}.rootindex;
    oidx{c} = csc_model.components{c}.offsetindex;
    root{c} = csc_model.rootfilters{ridx{c}}.w;
    root_pca{c} = csc_model.rootfilters{ridx{c}}.wpca;
    offset{c} = csc_model.offsets{oidx{c}}.w;
    loc{c} = csc_model.loc{c}.w;
    rsize{c} = [size(root{c},1) size(root{c},2)];
    numparts{c} = length(csc_model.components{c}.parts);
    for j = 1:numparts{c}
        pidx{c,j} = csc_model.components{c}.parts{j}.partindex;
        didx{c,j} = csc_model.components{c}.parts{j}.defindex;
        part{c,j} = csc_model.partfilters{pidx{c,j}}.w;
        part_pca{c,j} = csc_model.partfilters{pidx{c,j}}.wpca;
        psize{c,j} = [size(part{c,j},1) size(part{c,j},2)];
        % reverse map from partfilter index to (component, part#)
        % rpidx{pidx{c,j}} = [c j];
    end
end

maxsizex = ceil(csc_model.maxsize(2));
maxsizey = ceil(csc_model.maxsize(1));

pca_rows = size(csc_model.pca_coeff, 1);
pca_cols = size(csc_model.pca_coeff, 2);

f = fopen(fname_out, 'wb');
fprintf(f, '<?xml version="1.0"?>\n');
fprintf(f, '<opencv_storage>\n');
fprintf(f, '<SBin>%d</SBin>\n', csc_model.sbin);
fprintf(f, '<NumComponents>%d</NumComponents>\n', csc_model.numcomponents);
fprintf(f, '<NumFeatures>%d</NumFeatures>\n', num_feat);
fprintf(f, '<Interval>%d</Interval>\n', csc_model.interval);
fprintf(f, '<MaxSizeX>%d</MaxSizeX>\n', maxsizex);
fprintf(f, '<MaxSizeY>%d</MaxSizeY>\n', maxsizey);
%the pca coeff
fprintf(f, '<PCAcoeff type_id="opencv-matrix">\n');
fprintf(f,  '\t<rows>%d</rows>\n', pca_rows);
fprintf(f,  '\t<cols>%d</cols>\n', pca_cols);
fprintf(f,  '\t<dt>d</dt>\n');
fprintf(f,  '\t<data>\n');
for i=1:pca_rows
    fprintf(f,  '\t');
    for j=1:pca_cols
        fprintf(f,  '%f ', csc_model.pca_coeff(i, j));
    end
    fprintf(f,  '\n');
end
fprintf(f,  '\t</data>\n');
fprintf(f, '</PCAcoeff>\n');
fprintf(f, '<PCADim>%d</PCADim>\n', pca_cols);
fprintf(f, '<ScoreThreshold>%.16f</ScoreThreshold>\n', csc_model.thresh);

fprintf(f, '<Bias>\n');
for c = 1:csc_model.numcomponents
    fprintf(f,  '%f ', offset{c});
end
fprintf(f, '\n</Bias>\n');

fprintf(f, '<RootFilters>\n');
for c = 1:csc_model.numcomponents
    rootfilter = root{c};
    rows = size(rootfilter,1);
    cols = size(rootfilter,2);
    depth = size(rootfilter,3);
    fprintf(f, '\t<_ type_id="opencv-matrix">\n');
    fprintf(f,  '\t<rows>%d</rows>\n', rows);
    fprintf(f,  '\t<cols>%d</cols>\n', cols*depth);
    fprintf(f,  '\t<dt>d</dt>\n');
    fprintf(f,  '\t<data>\n');
    for i=1:rows
        fprintf(f,  '\t');
        for j=1:cols
            for k=1:depth
                fprintf(f,  '%f ', rootfilter(i, j, k));
            end
        end
        fprintf(f,  '\n');
    end
    fprintf(f,  '\t</data>\n');
    fprintf(f, '\t</_>\n');
end
fprintf(f, '</RootFilters>\n');

fprintf(f, '<RootPCAFilters>\n');
for c = 1:csc_model.numcomponents
    rootfilter_pca = root_pca{c};
    rows = size(rootfilter_pca,1);
    cols = size(rootfilter_pca,2);
    depth = size(rootfilter_pca,3);
    fprintf(f, '\t<_ type_id="opencv-matrix">\n');
    fprintf(f,  '\t<rows>%d</rows>\n', rows);
    fprintf(f,  '\t<cols>%d</cols>\n', cols*depth);
    fprintf(f,  '\t<dt>d</dt>\n');
    fprintf(f,  '\t<data>\n');
    for i=1:rows
        fprintf(f,  '\t');
        for j=1:cols
            for k=1:depth
                fprintf(f,  '%f ', rootfilter_pca(i, j, k));
            end
        end
        fprintf(f,  '\n');
    end
    fprintf(f,  '\t</data>\n');
    fprintf(f, '\t</_>\n');
end
fprintf(f, '</RootPCAFilters>\n');

fprintf(f, '<PartFilters>\n');
for c = 1:csc_model.numcomponents
    for p=1:numparts{c}
        partfilter = part{c,p};
        rows = size(partfilter,1);
        cols = size(partfilter,2);
        depth = size(partfilter,3);
        fprintf(f, '\t<_ type_id="opencv-matrix">\n');
        fprintf(f,  '\t<rows>%d</rows>\n', rows);
        fprintf(f,  '\t<cols>%d</cols>\n', cols*depth);
        fprintf(f,  '\t<dt>d</dt>\n');
        fprintf(f,  '\t<data>\n');
        for i=1:rows
            fprintf(f,  '\t');
            for j=1:cols
                for k=1:depth
                    fprintf(f,  '%f ', partfilter(i, j, k));
                end
            end
            fprintf(f,  '\n');
        end
        fprintf(f,  '\t</data>\n');
        fprintf(f, '\t</_>\n');
    end
end
fprintf(f, '</PartFilters>\n');

fprintf(f, '<PartPCAFilters>\n');
for c = 1:csc_model.numcomponents
    for p=1:numparts{c}
        partfilter = part_pca{c,p};
        rows = size(partfilter,1);
        cols = size(partfilter,2);
        depth = size(partfilter,3);
        fprintf(f, '\t<_ type_id="opencv-matrix">\n');
        fprintf(f,  '\t<rows>%d</rows>\n', rows);
        fprintf(f,  '\t<cols>%d</cols>\n', cols*depth);
        fprintf(f,  '\t<dt>d</dt>\n');
        fprintf(f,  '\t<data>\n');
        for i=1:rows
            fprintf(f,  '\t');
            for j=1:cols
                for k=1:depth
                    fprintf(f,  '%f ', partfilter(i, j, k));
                end
            end
            fprintf(f,  '\n');
        end
        fprintf(f,  '\t</data>\n');
        fprintf(f, '\t</_>\n');
    end
end
fprintf(f, '</PartPCAFilters>\n');

fprintf(f, '<PrunThreshold>\n');
for c = 1:csc_model.numcomponents
    fprintf(f, '\t<_>\n');
    fprintf(f,  '\t');
    t = csc_model.cascade.t{ridx{c}};
    for j=1:length(t)
        fprintf(f,  '%f ', t(j));
    end
    fprintf(f, '\n\t</_>\n');
end
fprintf(f, '</PrunThreshold>\n');

fprintf(f, '<Anchor>\n');
for c = 1:csc_model.numcomponents
    for p=1:numparts{c}
        fprintf(f, '\t<_>\n');
        fprintf(f,  '\t');
        anchor = csc_model.defs{didx{c,p}}.anchor;
        for j=1:length(anchor)
            fprintf(f,  '%f ', anchor(j));
        end
        fprintf(f, '\n\t</_>\n');
    end
end
fprintf(f, '</Anchor>\n');

fprintf(f, '<Deformation>\n');
for c = 1:csc_model.numcomponents
    for p=1:numparts{c}
        fprintf(f, '\t<_>\n');
        fprintf(f,  '\t');
        def = csc_model.defs{didx{c,p}}.w;
        for j=1:length(def)
            fprintf(f,  '%f ', def(j));
        end
        fprintf(f, '\n\t</_>\n');
    end
end
fprintf(f, '</Deformation>\n');

fprintf(f, '<NumParts>\n');
for c = 1:csc_model.numcomponents
    fprintf(f,  '%f ', numparts{c});
end
fprintf(f, '</NumParts>\n');

fprintf(f, '<PartOrder>\n');
for c = 1:csc_model.numcomponents
    fprintf(f, '\t<_>\n');
    fprintf(f,  '\t');
    order = csc_model.cascade.order{c};
    for i=1:length(order)
        fprintf(f,  '%f ', order(i));
    end
    fprintf(f, '\n\t</_>\n');
end
fprintf(f, '</PartOrder>\n');

fprintf(f, '<LocationWeight>\n');
for c = 1:csc_model.numcomponents
    fprintf(f, '\t<_>\n');
    fprintf(f,  '\t');
    loc_w = loc{c};
    for i=1:length(loc_w)
        fprintf(f,  '%f ', loc_w(i));
    end
    fprintf(f, '\n\t</_>\n');
end
fprintf(f, '</LocationWeight>\n');
fprintf(f, '</opencv_storage>');
fclose(f);