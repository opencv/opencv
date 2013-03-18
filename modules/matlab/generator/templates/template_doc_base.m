% {{ fun.name | upper }} {{ fun.doc.short_desc }}
{{ fun.doc.long_desc | comment('%',80) }}
%
% See also: {{ fun.doc.see_also }} 
%
% Copyright {{ time.strftime("%Y", localtime()) }} The OpenCV Foundation
