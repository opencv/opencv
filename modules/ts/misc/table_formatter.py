import sys, re, os.path, cgi, stat
from optparse import OptionParser
from color import getColorizer

class tblCell(object):
    def __init__(self, text, value = None, props = None):
        self.text = text
        self.value = value
        self.props = props

class tblColumn(object):
    def __init__(self, caption, title = None, props = None):
        self.text = caption
        self.title = title
        self.props = props
        
class tblRow(object):
    def __init__(self, colsNum, props = None):
        self.cells = [None] * colsNum
        self.props = props
        
def htmlEncode(str):
    return '<br/>'.join([cgi.escape(s) for s in str])

class table(object):
    def_align = "left"
    def_valign = "middle"
    def_color = None
    def_colspan = 1
    def_rowspan = 1
    def_bold = False
    def_italic = False
    def_text="-"

    def __init__(self, caption = None):
        self.columns = {}
        self.rows = []
        self.ridx = -1;
        self.caption = caption
        pass

    def newRow(self, **properties):
        if len(self.rows) - 1 == self.ridx:
            self.rows.append(tblRow(len(self.columns), properties))
        else:
            self.rows[ridx + 1].props = properties
        self.ridx += 1
    
    def trimLastRow(self):
        if self.rows:
            self.rows.pop()
        if self.ridx >= len(self.rows):
            self.ridx = len(self.rows) - 1

    def newColumn(self, name, caption, title = None, **properties):
        if name in self.columns:
            index = self.columns[name].index
        else:
            index = len(self.columns)
        if isinstance(caption, tblColumn):
            caption.index = index
            self.columns[name] = caption
            return caption
        else:
            col = tblColumn(caption, title, properties)
            col.index = index
            self.columns[name] = col
            return col

    def getColumn(self, name):
        if isinstance(name, str):
            return self.columns.get(name, None)
        else:
            vals = [v for v in self.columns.values() if v.index == name]
            if vals:
                return vals[0]
        return None

    def newCell(self, col_name, text, value = None, **properties):
        if self.ridx < 0:
            self.newRow()
        col = self.getColumn(col_name)
        row = self.rows[self.ridx]
        if not col:
            return None
        if isinstance(text, tblCell):
            cl = text
        else:
            cl = tblCell(text, value, properties)
        row.cells[col.index] = cl
        return cl
    
    def layoutTable(self):
        columns = self.columns.values()
        columns.sort(key=lambda c: c.index)
        
        colspanned = []
        rowspanned = []
        
        self.headerHeight = 1
        rowsToAppend = 0
        
        for col in columns:
            self.measureCell(col)
            if col.height > self.headerHeight:
                self.headerHeight = col.height
            col.minwidth = col.width
            col.line = None
        
        for r in range(len(self.rows)):
            row = self.rows[r]
            row.minheight = 1
            for i in range(len(row.cells)):
                cell = row.cells[i]
                if row.cells[i] is None:
                    continue
                cell.line = None
                self.measureCell(cell)
                colspan = int(self.getValue("colspan", cell))
                rowspan = int(self.getValue("rowspan", cell))
                if colspan > 1:
                    colspanned.append((r,i))
                    if i + colspan > len(columns):
                        colspan = len(columns) - i
                    cell.colspan = colspan
                    #clear spanned cells
                    for j in range(i+1, min(len(row.cells), i + colspan)):
                        row.cells[j] = None
                elif columns[i].minwidth < cell.width:
                    columns[i].minwidth = cell.width
                if rowspan > 1:
                    rowspanned.append((r,i))
                    rowsToAppend2 = r + colspan - len(self.rows)
                    if rowsToAppend2 > rowsToAppend:
                        rowsToAppend = rowsToAppend2
                    cell.rowspan = rowspan
                    #clear spanned cells
                    for j in range(r+1, min(len(self.rows), r + rowspan)):
                        if len(self.rows[j].cells) > i:
                            self.rows[j].cells[i] = None
                elif row.minheight < cell.height:
                    row.minheight = cell.height
                    
        self.ridx = len(self.rows) - 1
        for r in range(rowsToAppend):
            self.newRow()
            self.rows[len(self.rows) - 1].minheight = 1
            
        while colspanned:
            colspanned_new = []
            for r, c in colspanned:
                cell = self.rows[r].cells[c]
                sum([col.minwidth for col in columns[c:c + cell.colspan]])
                cell.awailable = sum([col.minwidth for col in columns[c:c + cell.colspan]]) + cell.colspan - 1
                if cell.awailable < cell.width:
                    colspanned_new.append((r,c))
            colspanned = colspanned_new
            if colspanned:
                r,c = colspanned[0]
                cell = self.rows[r].cells[c]
                cols = columns[c:c + cell.colspan]
                total = cell.awailable - cell.colspan + 1
                budget = cell.width - cell.awailable
                spent = 0
                s = 0
                for col in cols:
                    s += col.minwidth
                    addition = s * budget / total - spent
                    spent += addition
                    col.minwidth += addition
        
        while rowspanned:
            rowspanned_new = []
            for r, c in rowspanned:
                cell = self.rows[r].cells[c]
                cell.awailable = sum([row.minheight for row in self.rows[r:r + cell.rowspan]])
                if cell.awailable < cell.height:
                    rowspanned_new.append((r,c))
            rowspanned = rowspanned_new
            if rowspanned:
                r,c = rowspanned[0]
                cell = self.rows[r].cells[c]
                rows = self.rows[r:r + cell.rowspan]
                total = cell.awailable
                budget = cell.height - cell.awailable
                spent = 0
                s = 0
                for row in rows:
                    s += row.minheight
                    addition = s * budget / total - spent
                    spent += addition
                    row.minheight += addition
                    
        return columns

    def measureCell(self, cell):
        text = self.getValue("text", cell)
        cell.text = self.reformatTextValue(text)
        cell.height = len(cell.text)
        cell.width = len(max(cell.text, key = lambda line: len(line)))
                
    def reformatTextValue(self, value):
        if isinstance(value, str):
            vstr = value
        elif isinstance(value, unicode):
            vstr = str(value)
        else:
            try:
                vstr = '\n'.join([str(v) for v in value])
            except TypeError:
                vstr = str(value)
        return vstr.splitlines()
    
    def adjustColWidth(self, cols, width):
        total = sum([c.minWidth for c in cols])
        if total + len(cols) - 1 >= width:
            return
        budget = width - len(cols) + 1 - total
        spent = 0
        s = 0
        for col in cols:
            s += col.minWidth
            addition = s * budget / total - spent
            spent += addition
            col.minWidth += addition

    def getValue(self, name, *elements):
        for el in elements:
            try:
                return getattr(el, name)
            except AttributeError:
                pass
            try:
                val = el.props[name]
                if val:
                    return val
            except AttributeError:
                pass
            except KeyError:
                pass
        try:
            return getattr(self.__class__, "def_" + name)
        except AttributeError:
            return None
        
    def consolePrintTable(self, out):
        columns = self.layoutTable()
        colrizer = getColorizer(out) 
        
        if self.caption:
            out.write("%s%s%s" % ( os.linesep,  os.linesep.join(self.reformatTextValue(self.caption)), os.linesep * 2))
        
        headerRow = tblRow(len(columns), {"align": "center", "valign": "top", "bold": True, "header": True})
        headerRow.cells = columns
        headerRow.minheight = self.headerHeight
        
        self.consolePrintRow2(colrizer, headerRow, columns)
        
        for i in range(0, len(self.rows)):
            self.consolePrintRow2(colrizer, i, columns)
            
    def consolePrintRow2(self, out, r, columns):
        if isinstance(r, tblRow):
            row = r
            r = -1
        else:
            row = self.rows[r]
        
        #evaluate initial values for line numbers
        i = 0
        while i < len(row.cells):
            cell = row.cells[i]
            colspan = self.getValue("colspan", cell)
            if cell is not None:
                cell.wspace = sum([col.minwidth for col in columns[i:i + colspan]]) + colspan - 1
                if cell.line is None:
                    if r < 0:
                        rows = [row]
                    else:
                        rows = self.rows[r:r + self.getValue("rowspan", cell)]
                    cell.line = self.evalLine(cell, rows, columns[i])
                    if len(rows) > 1:
                        for rw in rows:
                            rw.cells[i] = cell
            i += colspan
            
        #print content
        for ln in range(row.minheight):
            i = 0
            while i < len(row.cells):
                if i > 0:
                    out.write(" ")
                cell = row.cells[i]
                column = columns[i]
                if cell is None:
                    out.write(" " * column.minwidth)
                    i += 1
                else:
                    self.consolePrintLine(cell, row, column, out)
                    i += self.getValue("colspan", cell)
            out.write(os.linesep)
                
    def consolePrintLine(self, cell, row, column, out):
        if cell.line < 0 or cell.line >= cell.height:
            line = ""
        else:
            line = cell.text[cell.line]
        width = cell.wspace
        align = self.getValue("align", ((None, cell)[isinstance(cell, tblCell)]), row, column)
        
        if align == "right":
            pattern = "%" + str(width) + "s"
        elif align == "center":
            pattern = "%" + str((width - len(line)) / 2 + len(line)) + "s" + " " * (width - len(line) - (width - len(line)) / 2)
        else:
            pattern = "%-" + str(width) + "s"
        
        out.write(pattern % line, color = self.getValue("color", cell, row, column))
        cell.line += 1
            
    def evalLine(self, cell, rows, column):
        height = cell.height
        valign = self.getValue("valign", cell, rows[0], column)
        space = sum([row.minheight for row in rows])
        if valign == "bottom":
            return height - space
        if valign == "middle":
            return (height - space + 1) / 2
        return 0
    
    def htmlPrintTable(self, out, embeedcss = False):
        columns = self.layoutTable()
        
        if embeedcss:
            out.write("<div style=\"font-family: Lucida Console, Courier New, Courier;font-size: 16px;color:#3e4758;\">\n<table style=\"background:none repeat scroll 0 0 #FFFFFF;border-collapse:collapse;font-family:'Lucida Sans Unicode','Lucida Grande',Sans-Serif;font-size:14px;margin:20px;text-align:left;width:480px;margin-left: auto;margin-right: auto;white-space:nowrap;\">\n")
        else:
            out.write("<div class=\"tableFormatter\">\n<table class=\"tbl\">\n")
        if self.caption:
            if embeedcss:
                out.write(" <caption style=\"font:italic 16px 'Trebuchet MS',Verdana,Arial,Helvetica,sans-serif;padding:0 0 5px;text-align:right;white-space:normal;\">%s</caption>\n" % htmlEncode(self.reformatTextValue(self.caption)))
            else:
                out.write(" <caption>%s</caption>\n" % htmlEncode(self.reformatTextValue(self.caption)))
        out.write(" <thead>\n")
        
        headerRow = tblRow(len(columns), {"align": "center", "valign": "top", "bold": True, "header": True})
        headerRow.cells = columns
        
        header_rows = [headerRow]
        header_rows.extend([row for row in self.rows if self.getValue("header")])
        last_row = header_rows[len(header_rows) - 1]
        
        for row in header_rows:
            out.write("  <tr>\n")
            for th in row.cells:
                align = self.getValue("align", ((None, th)[isinstance(th, tblCell)]), row, row)
                valign = self.getValue("valign", th, row)
                cssclass = self.getValue("cssclass", th)
                attr = ""
                if align:
                    attr += " align=\"%s\"" % align
                if valign:
                    attr += " valign=\"%s\"" % valign
                if cssclass:
                    attr += " class=\"%s\"" % cssclass
                css = ""
                if embeedcss:
                    css = " style=\"border:none;color:#003399;font-size:16px;font-weight:normal;white-space:nowrap;padding:3px 10px;\""
                    if row == last_row:
                        css = css[:-1] + "padding-bottom:5px;\""
                out.write("   <th%s%s>\n" % (attr, css))
                if th is not None:
                    out.write("    %s\n" % htmlEncode(th.text))
                out.write("   </th>\n")
            out.write("  </tr>\n")
            
        out.write(" </thead>\n <tbody>\n")
        
        rows = [row for row in self.rows if not self.getValue("header")]
        for r in range(len(rows)):
            row = rows[r]
            out.write("  <tr>\n")
            i = 0
            while i < len(row.cells):
                column = columns[i] 
                td = row.cells[i]
                if isinstance(td, int):
                    i += td
                    continue
                colspan = self.getValue("colspan", td)
                rowspan = self.getValue("rowspan", td)
                align = self.getValue("align", td, row, column)
                valign = self.getValue("valign", td, row, column)
                color = self.getValue("color", td, row, column)
                bold = self.getValue("bold", td, row, column)
                italic = self.getValue("italic", td, row, column)
                style = ""
                attr = ""
                if color:
                    style += "color:%s;" % color
                if bold:
                    style += "font-weight: bold;"
                if italic:
                    style += "font-style: italic;"
                if align and align != "left":
                    attr += " align=\"%s\"" % align
                if valign and valign != "middle":
                    attr += " valign=\"%s\"" % valign
                if colspan > 1:
                    attr += " colspan=\"%s\"" % colspan
                if rowspan > 1:
                    attr += " rowspan=\"%s\"" % rowspan
                    for q in range(r+1, min(r+rowspan, len(rows))):
                        rows[q].cells[i] = colspan
                if style:
                    attr += " style=\"%s\"" % style
                css = ""
                if embeedcss:
                    css = " style=\"border:none;border-bottom:1px solid #CCCCCC;color:#666699;padding:6px 8px;white-space:nowrap;\""
                    if r == 0:
                        css = css[:-1] + "border-top:2px solid #6678B1;\""
                out.write("   <td%s%s>\n" % (attr, css))
                if th is not None:
                    out.write("    %s\n" % htmlEncode(td.text))
                out.write("   </td>\n")
                i += colspan
            out.write("  </tr>\n")
            
        out.write(" </tbody>\n</table>\n</div>\n")
        
def htmlPrintHeader(out, title = None):
    if title:
        titletag = "<title>%s</title>\n" % htmlEncode([str(title)])
    else:
        titletag = ""
    out.write("""<!DOCTYPE HTML>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=us-ascii">
%s<style type="text/css">
html, body {font-family: Lucida Console, Courier New, Courier;font-size: 16px;color:#3e4758;}
.tbl{background:none repeat scroll 0 0 #FFFFFF;border-collapse:collapse;font-family:"Lucida Sans Unicode","Lucida Grande",Sans-Serif;font-size:14px;margin:20px;text-align:left;width:480px;margin-left: auto;margin-right: auto;white-space:nowrap;}
.tbl span{display:block;white-space:nowrap;}
.tbl thead tr:last-child th {padding-bottom:5px;}
.tbl tbody tr:first-child td {border-top:2px solid #6678B1;}
.tbl th{border:none;color:#003399;font-size:16px;font-weight:normal;white-space:nowrap;padding:3px 10px;}
.tbl td{border:none;border-bottom:1px solid #CCCCCC;color:#666699;padding:6px 8px;white-space:nowrap;}
.tbl tbody tr:hover td{color:#000099;}
.tbl caption{font:italic 16px "Trebuchet MS",Verdana,Arial,Helvetica,sans-serif;padding:0 0 5px;text-align:right;white-space:normal;}
</style>
<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/1.6.4/jquery.min.js"></script>
<script type="text/javascript">
function abs(val) { return val < 0 ? -val : val }
$(function(){
  //generate filter rows
  $("div.tableFormatter table.tbl").each(function(tblIdx, tbl) {
    var head = $("thead", tbl)
    var filters = $("<tr></tr>")
    var hasAny = false
    $("tr:first th", head).each(function(colIdx, col) {
      col = $(col)
      var cell
      var id = "t" + tblIdx + "r" + colIdx
      if (col.hasClass("col_name")){
        cell = $("<th><input id='" + id + "' name='" + id + "' type='text' style='width:100%%' class='filter_col_name' title='Regular expression for name filtering'></input></th>")
        hasAny = true
      }
      else if (col.hasClass("col_rel")){
        cell = $("<th><input id='" + id + "' name='" + id + "' type='text' style='width:100%%' class='filter_col_rel' title='Filter all lines with speedup less than &lt;value&gt;'></input></th>")
        hasAny = true
      }
      else
        cell = $("<th></th>")
      cell.appendTo(filters)
    })

   if (hasAny){
     $(tbl).wrap("<form id='form_t" + tblIdx + "' method='get' action=''></form>")
     $("<input it='test' type='submit' value='Apply Filters' style='margin-left:10px;'></input>")
       .appendTo($("th:last", filters.appendTo(head)))
   }
  })

  //get filter values
  var vars = []
  var hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&')
  for(var i = 0; i < hashes.length; ++i)
  {
     hash = hashes[i].split('=')
     vars.push(decodeURIComponent(hash[0]))
     vars[decodeURIComponent(hash[0])] = decodeURIComponent(hash[1]);
  }

  //set filter values
  for(var i = 0; i < vars.length; ++i)
     $("#" + vars[i]).val(vars[vars[i]])

  //apply filters
  $("div.tableFormatter table.tbl").each(function(tblIdx, tbl) {
      filters = $("input:text", tbl)
      var predicate = function(row) {return true;}
      var empty = true
      $.each($("input:text", tbl), function(i, flt) {
         flt = $(flt)
         var val = flt.val()
         var pred = predicate;
         if(val) {
           empty = false
           var colIdx = parseInt(flt.attr("id").slice(flt.attr("id").indexOf('r') + 1))
           if(flt.hasClass("filter_col_name")) {
              var re = new RegExp(val);
              predicate = function(row) {
                if (re.exec($(row.get(colIdx)).text()) == null)
                  return false
                return pred(row)
	      }
           } else if(flt.hasClass("filter_col_rel")) {
              var percent = val.indexOf('.') < 0 ? parseInt(val)*0.01 : parseFloat(val)
              predicate = function(row) {
                if (abs(parseFloat($(row.get(colIdx)).text()) - 1) < percent)
                  return false
                return pred(row)
	      }
           }
         }
      });
      if (!empty){
         $("tbody tr", tbl).each(function (i, tbl_row) {
            if(!predicate($("td", tbl_row)))
               $(tbl_row).remove()
         })
      }
  })
})
</script>
</head>
<body>
""" % titletag)
    
def htmlPrintFooter(out):
    out.write("</body>\n</html>")
    
def getStdoutFilename():
    try:
        if os.name == "nt":
            import msvcrt, ctypes
            handle = msvcrt.get_osfhandle(sys.stdout.fileno())
            size = ctypes.c_ulong(1024)
            nameBuffer = ctypes.create_string_buffer(size.value)
            ctypes.windll.kernel32.GetFinalPathNameByHandleA(handle, nameBuffer, size, 4)
            return nameBuffer.value
        else:
            return os.readlink('/proc/self/fd/1')
    except:
        return ""
    
def detectHtmlOutputType(requestedType):
    if requestedType == "txt":
        return False
    elif requestedType in ["html", "moinwiki"]:
        return True
    else:
        if sys.stdout.isatty():
            return False
        else:
            outname = getStdoutFilename()
            if outname:
                if outname.endswith(".htm") or outname.endswith(".html"):
                    return True
                else:
                    return False
            else:
                return False 
            
def getRelativeVal(test, test0, metric):
    if not test or not test0:
        return None
    val0 = test0.get(metric, "s")
    if not val0 or val0 == 0:
        return None
    val =  test.get(metric, "s")
    if not val:
        return None
    return float(val)/val0

        
metrix_table = \
{
    "name": ("Name of Test", lambda test,test0,units: str(test)),
    
    "samples": ("Number of\ncollected samples", lambda test,test0,units: test.get("samples", units)),
    "outliers": ("Number of\noutliers", lambda test,test0,units: test.get("outliers", units)),
    
    "gmean": ("Geometric mean", lambda test,test0,units: test.get("gmean", units)),
    "mean": ("Mean", lambda test,test0,units: test.get("mean", units)),
    "min": ("Min", lambda test,test0,units: test.get("min", units)),
    "median": ("Median", lambda test,test0,units: test.get("median", units)),
    "stddev": ("Standard deviation", lambda test,test0,units: test.get("stddev", units)),
    "gstddev": ("Standard deviation of Ln(time)", lambda test,test0,units: test.get("gstddev")),
    
    "gmean%": ("Geometric mean (relative)", lambda test,test0,units: getRelativeVal(test, test0, "gmean")),
    "mean%": ("Mean (relative)", lambda test,test0,units: getRelativeVal(test, test0, "mean")),
    "min%": ("Min (relative)", lambda test,test0,units: getRelativeVal(test, test0, "min")),
    "median%": ("Median (relative)", lambda test,test0,units: getRelativeVal(test, test0, "median")),
    "stddev%": ("Standard deviation (relative)", lambda test,test0,units: getRelativeVal(test, test0, "stddev")),
    "gstddev%": ("Standard deviation of Ln(time) (relative)", lambda test,test0,units: getRelativeVal(test, test0, "gstddev")),
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage:\n", os.path.basename(sys.argv[0]), "<log_name>.xml"
        exit(0)
        
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="format", help="output results in text format (can be 'txt', 'html' or 'auto' - default)", metavar="FMT", default="auto")
    parser.add_option("-m", "--metric", dest="metric", help="output metric", metavar="NAME", default="gmean")
    parser.add_option("-u", "--units", dest="units", help="units for output values (s, ms (default), mks, ns or ticks)", metavar="UNITS", default="ms")
    (options, args) = parser.parse_args()
    
    options.generateHtml = detectHtmlOutputType(options.format)
    if options.metric not in metrix_table:
        options.metric = "gmean"
    
    #print options
    #print args

#    tbl = table()
#    tbl.newColumn("first", "qqqq", align = "left")
#    tbl.newColumn("second", "wwww\nz\nx\n")
#    tbl.newColumn("third", "wwasdas")
#
#    tbl.newCell(0, "ccc111", align = "right")
#    tbl.newCell(1, "dddd1")
#    tbl.newCell(2, "8768756754")
#    tbl.newRow()
#    tbl.newCell(0, "1\n2\n3\n4\n5\n6\n7", align = "center", colspan = 2, rowspan = 2)
#    tbl.newCell(2, "xxx\nqqq", align = "center", colspan = 1, valign = "middle")
#    tbl.newRow()
#    tbl.newCell(2, "+", align = "center", colspan = 1, valign = "middle")
#    tbl.newRow()
#    tbl.newCell(0, "vcvvbasdsadassdasdasv", align = "right", colspan = 2)
#    tbl.newCell(2, "dddd1")
#    tbl.newRow()
#    tbl.newCell(0, "vcvvbv")
#    tbl.newCell(1, "3445324", align = "right")
#    tbl.newCell(2, None)
#    tbl.newCell(1, "0000")
#    if sys.stdout.isatty():
#        tbl.consolePrintTable(sys.stdout)
#    else:
#        htmlPrintHeader(sys.stdout)
#        tbl.htmlPrintTable(sys.stdout)
#        htmlPrintFooter(sys.stdout)

    import testlog_parser
    
    if options.generateHtml:
        htmlPrintHeader(sys.stdout, "Tables demo")
        
    getter = metrix_table[options.metric][1]

    for arg in args:
        tests = testlog_parser.parseLogFile(arg)
        tbl = table(arg)
        tbl.newColumn("name", "Name of Test", align = "left")
        tbl.newColumn("value", metrix_table[options.metric][0], align = "center", bold = "true")
        
        for t in sorted(tests):
            tbl.newRow()
            tbl.newCell("name", str(t))
            
            status = t.get("status")
            if status != "run":
                tbl.newCell("value", status)
            else:
                val = getter(t, None, options.units)
                if val:
                    if options.metric.endswith("%"):
                        tbl.newCell("value", "%.2f" % val, val)
                    else:
                        tbl.newCell("value", "%.3f %s" % (val, options.units), val)
                else:
                    tbl.newCell("value", "-")
        
        if options.generateHtml:
            tbl.htmlPrintTable(sys.stdout)
        else:
            tbl.consolePrintTable(sys.stdout)
            
    if options.generateHtml:
        htmlPrintFooter(sys.stdout)
