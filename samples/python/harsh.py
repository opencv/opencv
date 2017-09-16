import random
import subprocess
max_authors = 1

comment_string = {'py': '#', 'sh': "#",  'cpp': '//'}


def get_authors(file_contents, ptype):
    Authors = []
    for line in file_contents.lower().splitlines():
        if line.startswith(comment_string[ptype]) and "copyright" in line:
            try:
                _, email = line.rsplit(" ", 1)
                if email.endswith('@bu.edu'):
                    Authors.append(email)
            except:
                pass
    return Authors


def progtype(program):
    _, program_type = program.split('.')
    return program_type

testwords = ['apple', 'orange', 'kiwi', 'banana',
             'strawberry', 'pineapple', 'rasberry']


def test_fourargspy(actualname):
    s = ""
    for n in [0, 3, 4, 5, 7]:
        words = random.sample(testwords, n)
        try:
            T = subprocess.run(['python', actualname, *words],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,timeout=2)
        except:
            s +="your script timed out."
            return s

        if T.stdout.decode() != ''.join([a+'\n' for a in words[:4]]):
            s += "Your stdout is not correct for {} arguments.\n".format(n)
        if T.stderr.decode() != ''.join([a+'\n' for a in words[4:]]):
            s += "Your stderr is not correct for {} arguments.\n".format(n)
    return s


def test_fourargscpp(actualname):
    s = ""
    compiledprogram = actualname[:-4]
    C = subprocess.run(['g++', actualname, '-o', compiledprogram],
                       stderr=subprocess.PIPE)
    if C.returncode:
        s = 'g++ found problems, as follows:\n'
        s += C.stderr.decode()
        return s
    for n in [0, 3, 4, 5, 7]:
        words = random.sample(testwords, n)
        try:
            T = subprocess.run([compiledprogram, *words], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,timeout=1)
        except:
            s +="your script timed out."
            return s
        if T.stdout.decode() != ''.join([a+'\n' for a in words[:4]]):
            s += "Your stdout is not correct for {} arguments.\n".format(n)
        if T.stderr.decode() != ''.join([a+'\n' for a in words[4:]]):
            s += "Your stderr is not correct for {} arguments.\n".format(n)
    return s


def test_fourargssh(actualname):
    s = ""
    try:
        T = subprocess.run([actualname], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, shell=True,timeout=5)
    except:
        s +="your script timed out."
        return s

    if len(T.stdout.decode().strip().split('\n')) != 14:
        s += 'your shell script is not correct (stdout problem)\n'
        s += 'You printed out:'+T.stdout.decode()
    if len(T.stderr.decode().strip().split('\n')) != 4:
        s += 'your shell script is not correct (stdout problem)\n'
    return s

programs = {'fourargs.py': test_fourargspy,
            'fourargs.cpp': test_fourargscpp,
            'fourargs.sh': test_fourargssh}


def analyse(program,actualprogramname=None):
    
    actualprogramname = actualprogramname or program

    s = 'Checking {} for EC602 submission.\n'.format(program)
    ptype = progtype(program)
    try:
        f = open(actualprogramname)
        contents = f.read()
        f.close()
    except:
        s += 'The program {} does not exist here.\n'.format(actualprogramname)
        return 'No file', s

    authors = get_authors(contents, ptype)
    s += 'authors       : {}\n'.format(" ".join(authors))

    if len(authors) > max_authors:
        s += "You have exceeded the maximum number of authors.\n"
        return 'Too many authors', s

    res = programs[program](actualprogramname)
    s += 'program check :'
    if res:
        s += " failed.\n"
        s += res
        return False, s
    else:
        s += " passed.\n"
        return "Pass", s

if __name__ == '__main__':
    for program in programs:
        summary, results = analyse(program)
        print(results)
