#!/usr/bin/env python

'''
Sample-launcher application.
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys

# local modules
from common import splitfn

# built-in modules
import webbrowser
from glob import glob
from subprocess import Popen

try:
    import tkinter as tk  # Python 3
    from tkinter.scrolledtext import ScrolledText
except ImportError:
    import Tkinter as tk  # Python 2
    from ScrolledText import ScrolledText


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
                self.samples[name] = fn

        for name in sorted(self.samples):
            demos_lb.insert(tk.END, name)

        demos_lb.bind('<<ListboxSelect>>', self.on_demo_select)

        self.cmd_entry = cmd_entry = tk.Entry(right)
        cmd_entry.bind('<Return>', self.on_run)
        run_btn = tk.Button(right, command=self.on_run, text='Run', width=8)

        self.text = text = ScrolledText(right, font=('arial', 12, 'normal'), width = 30, wrap='word')
        self.linker = _linker = LinkManager(text, self.on_link)
        self.text.tag_config("header1", font=('arial', 14, 'bold'))
        self.text.tag_config("header2", font=('arial', 12, 'bold'))
        text.config(state='disabled')

        text.pack(fill='both', expand=1, side=tk.BOTTOM)
        cmd_entry.pack(fill='x', side='left' , expand=1)
        run_btn.pack()

    def on_link(self, url):
        print(url)
        webbrowser.open(url)

    def on_demo_select(self, evt):
        name = self.demos_lb.get( self.demos_lb.curselection()[0] )
        fn = self.samples[name]

        descr = ""
        try:
            if sys.version_info[0] > 2:
                # Python 3.x
                module_globals = {}
                module_locals = {}
                with open(fn, 'r') as f:
                    module_code = f.read()
                exec(compile(module_code, fn, 'exec'), module_globals, module_locals)
                descr = module_locals.get('__doc__', 'no-description')
            else:
                # Python 2
                module_globals = {}
                execfile(fn, module_globals)  # noqa: F821
                descr = module_globals.get('__doc__', 'no-description')
        except Exception as e:
            descr = str(e)

        self.linker.reset()
        self.text.config(state='normal')
        self.text.delete(1.0, tk.END)
        self.format_text(descr)
        self.text.config(state='disabled')

        self.cmd_entry.delete(0, tk.END)
        self.cmd_entry.insert(0, fn)

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

    def match_text(self, pattern, tag_proc, regexp=True):
        text = self.text
        text.mark_set('matchPos', '1.0')
        count = tk.IntVar()
        while True:
            match_index = text.search(pattern, 'matchPos', count=count, regexp=regexp, stopindex='end')
            if not match_index:
                break
            end_index = text.index( "%s+%sc" % (match_index, count.get()) )
            text.mark_set('matchPos', end_index)
            if callable(tag_proc):
                tag_proc(match_index, end_index, text.get(match_index, end_index))
            else:
                text.tag_add(tag_proc, match_index, end_index)

    def on_run(self, *args):
        cmd = self.cmd_entry.get()
        print('running:', cmd)
        Popen(sys.executable + ' ' + cmd, shell=True)

    def run(self):
        tk.mainloop()


if __name__ == '__main__':
    App().run()
