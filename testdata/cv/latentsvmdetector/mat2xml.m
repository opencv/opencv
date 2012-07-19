function [] = mat2xml(fname_in, fname_out)
load(fname_in);
num_feat = 31;
rootfilters = [];
for i = 1:length(model.rootfilters)
  rootfilters{i} = model.rootfilters{i}.w;
end
partfilters = [];
for i = 1:length(model.partfilters)
  partfilters{i} = model.partfilters{i}.w;
end
for c = 1:model.numcomponents
  ridx{c} = model.components{c}.rootindex;
  oidx{c} = model.components{c}.offsetindex;
  root{c} = model.rootfilters{ridx{c}}.w;
  rsize{c} = [size(root{c},1) size(root{c},2)];
  numparts{c} = length(model.components{c}.parts);
  for j = 1:numparts{c}
    pidx{c,j} = model.components{c}.parts{j}.partindex;
    didx{c,j} = model.components{c}.parts{j}.defindex;
    part{c,j} = model.partfilters{pidx{c,j}}.w;
    psize{c,j} = [size(part{c,j},1) size(part{c,j},2)];
    % reverse map from partfilter index to (component, part#)
    % rpidx{pidx{c,j}} = [c j];
  end
end

f = fopen(fname_out, 'wb');
fprintf(f, '<Model>\n');
fprintf(f, '\t<!-- Number of components -->\n');
fprintf(f, '\t<NumComponents>%d</NumComponents>\n', model.numcomponents);
fprintf(f, '\t<!-- Number of features -->\n');
fprintf(f, '\t<P>%d</P>\n', num_feat);
fprintf(f, '\t<!-- Score threshold -->\n');
fprintf(f, '\t<ScoreThreshold>%.16f</ScoreThreshold>\n', model.thresh);
for c = 1:model.numcomponents
    fprintf(f, '\t<Component>\n');
    fprintf(f, '\t\t<!-- Root filter description -->\n');
    fprintf(f, '\t\t<RootFilter>\n');
    fprintf(f, '\t\t\t<!-- Dimensions -->\n');
    rootfilter = root{c};
    fprintf(f, '\t\t\t<sizeX>%d</sizeX>\n', rsize{c}(2));
    fprintf(f, '\t\t\t<sizeY>%d</sizeY>\n', rsize{c}(1));
    fprintf(f, '\t\t\t<!-- Weights (binary representation) -->\n');
    fprintf(f, '\t\t\t<Weights>');
    for jj = 1:rsize{c}(1)
        for ii = 1:rsize{c}(2)
            for kk = 1:num_feat
                fwrite(f, rootfilter(jj, ii, kk), 'double');
            end
        end
    end
    fprintf(f, '\t\t\t</Weights>\n');
    fprintf(f, '\t\t\t<!-- Linear term in score function -->\n');
    fprintf(f, '\t\t\t<LinearTerm>%.16f</LinearTerm>\n', model.offsets{1,c}.w);
    fprintf(f, '\t\t</RootFilter>\n\n');
    fprintf(f, '\t\t<!-- Part filters description -->\n');
    fprintf(f, '\t\t<PartFilters>\n');
    fprintf(f, '\t\t\t<NumPartFilters>%d</NumPartFilters>\n', numparts{c});

    for j=1:numparts{c}
        fprintf(f, '\t\t\t<!-- Part filter ¹%d description -->\n', j);
        fprintf(f, '\t\t\t<PartFilter>\n');
        partfilter = part{c,j};
        anchor = model.defs{didx{c,j}}.anchor;
        def = model.defs{didx{c,j}}.w;
        
        fprintf(f, '\t\t\t\t<!-- Dimensions -->\n');
        fprintf(f, '\t\t\t\t<sizeX>%d</sizeX>\n', psize{c,j}(2));
        fprintf(f, '\t\t\t\t<sizeY>%d</sizeY>\n', psize{c,j}(1));
        fprintf(f, '\t\t\t\t<!-- Weights (binary representation) -->\n');
        fprintf(f, '\t\t\t\t<Weights>');
        for jj = 1:psize{c,j}(1)
            for ii = 1:psize{c,j}(2)
                for kk = 1:num_feat
                    fwrite(f, partfilter(jj, ii, kk), 'double');
                end
            end
        end
        fprintf(f, '\t\t\t\t</Weights>\n');
        fprintf(f, '\t\t\t\t<!-- Part filter offset -->\n');
        fprintf(f, '\t\t\t\t<V>\n');
        fprintf(f, '\t\t\t\t\t<Vx>%d</Vx>\n', anchor(1));
        fprintf(f, '\t\t\t\t\t<Vy>%d</Vy>\n', anchor(2));
        fprintf(f, '\t\t\t\t</V>\n');
        fprintf(f, '\t\t\t\t<!-- Quadratic penalty function coefficients -->\n');
        fprintf(f, '\t\t\t\t<Penalty>\n');
        fprintf(f, '\t\t\t\t\t<dx>%.16f</dx>\n', def(2));
        fprintf(f, '\t\t\t\t\t<dy>%.16f</dy>\n', def(4));
        fprintf(f, '\t\t\t\t\t<dxx>%.16f</dxx>\n', def(1));
        fprintf(f, '\t\t\t\t\t<dyy>%.16f</dyy>\n', def(3));
        fprintf(f, '\t\t\t\t</Penalty>\n');
         fprintf(f, '\t\t\t</PartFilter>\n');
    end
    fprintf(f, '\t\t</PartFilters>\n');
    fprintf(f, '\t</Component>\n');
end
fprintf(f, '</Model>');
fclose(f);
