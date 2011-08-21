import Tkinter as tk
from ScrolledText import ScrolledText
from glob import glob
from common import splitfn
import webbrowser

#from IPython.Shell import IPShellEmbed
#ipshell = IPShellEmbed()

exclude_list = ['demo', 'common']

class LinkManager:
    def __init__(self, text, url_callback = None):
        self.text = text
        self.text.tag_config("link", foreground="blue", underline=1)
        self.text.tag_bind("link", "<Enter>", self._enter)
        self.text.tag_bind("link", "<Leave>", self._leave)
        self.text.tag_bind("link", "<Button-1>", self._click)

        self.url_callback = url_callback
        self.reset()

    def reset(self):
        self.links = {}
    def add(self, action):
        # add an action to the manager.  returns tags to use in
        # associated text widget
        tag = "link-%d" % len(self.links)
        self.links[tag] = action
        return "link", tag

    def _enter(self, event):
        self.text.config(cursor="hand2")
    def _leave(self, event):
        self.text.config(cursor="")
    def _click(self, event):
        for tag in self.text.tag_names(tk.CURRENT):
            if tag.startswith("link-"):
                proc = self.links[tag]
                if callable(proc):
                    proc()
                else:
                    if self.url_callback:
                        self.url_callback(proc)

class App:
    def __init__(self):
        root = tk.Tk()
        root.title('OpenCV Demo')

        self.win = win = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        self.win.pack(fill=tk.BOTH, expand=1)

        left = tk.Frame(win)
        right = tk.Frame(win)                 
        win.add(left)
        win.add(right)
                            
        scrollbar = tk.Scrollbar(left, orient=tk.VERTICAL)
        self.demos_lb = demos_lb = tk.Listbox(left, yscrollcommand=scrollbar.set)
        scrollbar.config(command=demos_lb.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        demos_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        self.samples = {}
        for fn in glob('*.py'):
            name = splitfn(fn)[1]
            if fn[0] != '_' and name not in exclude_list:
                demos_lb.insert(tk.END, name)
                self.samples[name] = fn
        demos_lb.bind('<<ListboxSelect>>', self.on_demo_select)

        self.text = text = ScrolledText(right, font=('arial', 12, 'normal'), width = 30, wrap='word')
        text.pack(fill='both', expand=1)
        self.linker = linker = LinkManager(text, self.on_link)

        self.text.tag_config("header1", font=('arial', 14, 'bold'))
        self.text.tag_config("header2", font=('arial', 12, 'bold'))

        text.config(state='disabled')

    def on_link(self, url):
        print url
        webbrowser.open(url)

    def on_demo_select(self, evt):
        name = self.demos_lb.get( self.demos_lb.curselection()[0] )
        fn = self.samples[name]
        loc = {}
        execfile(fn, loc)
        descr = loc.get('__doc__', 'no-description')
        
        self.linker.reset()
        self.text.config(state='normal')
        self.text.delete(1.0, tk.END)
        self.format_text(descr)
        self.text.config(state='disabled')

    def format_text(self, s):
        text = self.text
        lines = s.splitlines()
        for i, s in enumerate(lines):
            s = s.rstrip()
            if i == 0 and not s:
                continue
            if s and s == '='*len(s):
                text.tag_add('header1', 'end-2l', 'end-1l')
            elif s and s == '-'*len(s):
                text.tag_add('header2', 'end-2l', 'end-1l')
            else:
                text.insert('end', s+'\n')
        
        def add_link(start, end, url):
            for tag in self.linker.add(url):
                text.tag_add(tag, start, end)
        self.match_text(r'http://\S+', add_link)

    def match_text(self, pattern, tag_proc):
        text = self.text
        text.mark_set('matchPos', '1.0')
        count = tk.IntVar()
        while True:
            match_index = text.search(pattern, 'matchPos', count=count, regexp=True, stopindex='end')
            if not match_index: break
            end_index = text.index( "%s+%sc" % (match_index, count.get()) )
            text.mark_set('matchPos', end_index)
            if callable(tag_proc):
                tag_proc(match_index, end_index, text.get(match_index, end_index))
            else:
                text.tag_add(tag_proc, match_index, end_index)

    def run(self):
        tk.mainloop()


if __name__ == '__main__':
    App().run()
