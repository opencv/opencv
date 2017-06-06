// memoryprofiler.js: Enables HEAP memory usage profiling in Emscripten applications.

// CONFIGURATION: If true, walks all allocated pointers at graphing time to print a detailed memory fragmentation map. If false, used
// memory is only graphed in one block (at the bottom of DYNAMIC memory space). Set this to false to improve performance at the expense of
// accuracy.
var MEMORYPROFILER_DETAILED_HEAP_USAGE = true;

// CONFIGURATION: Allocations of memory blocks larger than this threshold will get their detailed callstack captured and logged at runtime.
// Warning: This can be extremely slow. Set to a very very large value like 1024*1024*1024*4 to disable.
var MEMORYPROFILER_TRACK_CALLSTACK_MIN_SIZE = 16*1024*1024;
if (typeof new Error().stack === 'undefined') { // Disable callstack tracking if stack information is not available in this browser (at least IE and Safari don't have this)
  MEMORYPROFILER_TRACK_CALLSTACK_MIN_SIZE = 1024*1024*1024*4;
}

// Controls whether all outstanding allocation are printed to html page by callstack.
var MEMORYPROFILER_ALLOC_STATS = false;
// If MEMORYPROFILER_ALLOC_STATS = true, then all callstacks that have recorded more than the following number of allocations will be printed to html page.
var MEMORYPROFILER_ALLOC_STATS_MIN_REPORTED = 100;
// Tracks data for the above.
var memoryprofiler_alloc_site_stats = {}; 
var memoryprofiler_alloc_site_ptrs = {};

// CONFIGURATION: If true, we hook into Runtime.stackAlloc to be able to catch better estimate of the maximum used STACK space.
// You might only ever want to set this to false for performance reasons. Since stack allocations may occur often, this might impact
// performance.
var MEMORYPROFILER_HOOK_STACKALLOC = true;

// CONFIGURATION: UI update interval in milliseconds:
var MEMORYPROFILER_UI_UPDATE_INTERVAL = 2000;

// Stores an associative array of records HEAP ptr -> size so that we can retrieve how much memory was freed in calls to 
// _free() and decrement the tracked usage accordingly.
// E.g. memoryprofiler_ptr_to_size[address] returns the size of the heap pointer starting at 'address'.
var memoryprofiler_ptr_to_size = {};

// Conceptually same as the above array, except this one tracks only pointers that were allocated during the application preRun step, which
// corresponds to the data added to the VFS with --preload-file.
var memoryprofiler_prerun_mallocs = {};
var memoryprofiler_page_prerun_is_finished = false; // Once set to true, preRun is finished and the above array is not touched anymore.

// Stores an associative array of records HEAP ptr -> function string name so that we can identify each allocated pointer 
// by the location in code the allocation occurred in.
var memoryprofiler_ptr_to_loc = {};

// Stores an associative array of accumulated amount of memory allocated per location.
// E.g. memoryprofiler_loc_to_size[callstack_function_name_string] returns the total number of allocated bytes performed from 'callstack_function_name_string'.
var memoryprofiler_loc_to_size = {};

// Grand total of memory currently allocated via malloc(). Decremented on free()s.
var memoryprofiler_total_mem_allocated = 0;

// The running count of the number of times malloc() and free() have been called in the app. Used to keep track of # of currently alive pointers.
// TODO: Perhaps in the future give a statistic of allocations per second to see how trashing memory usage is.
var memoryprofiler_num_allocs = 0;
var memoryprofiler_num_frees = 0;

var memoryprofiler_stacktop_watermark = 0;

// Converts number f to string with at most two decimals. 
function truncDec(f) {
  var str = f.toFixed(2);
  if (str.indexOf('.00', str.length-3) !== -1) {
    return str.substr(0, str.length-3);
  } else if (str.indexOf('0', str.length-1) !== -1) {
    return str.substr(0, str.length-1);
  } else {
    return str;
  }
}

// Converts a number of bytes pretty-formatted as a string.
function formatBytes(bytes) {
  if (bytes >= 1000*1024*1024) {
    return truncDec(bytes/(1024*1024*1024)) + ' GB';
  } else if (bytes >= 1000*1024) {
    return truncDec(bytes/(1024*1024)) + ' MB';
  } else if (bytes >= 1000) {
    return truncDec(bytes/1024) + ' KB';
  } else {
    return truncDec(bytes) + ' B';
  }
}

// Installs startup hook and periodic UI update timer.
function memoryprofiler_add_hooks() {
  var prevMalloc = _malloc;
  function hookedMalloc(size) {
    // Gather global stats.
    memoryprofiler_total_mem_allocated += size;
    ++memoryprofiler_num_allocs;
    memoryprofiler_stacktop_watermark = Math.max(memoryprofiler_stacktop_watermark, STACKTOP);
    
    // Call the real allocator.
    var ptr = prevMalloc(size);
    if (!ptr) {
      return 0;
    }
    
    // Remember the size of the allocated block to know how much will be _free()d later.
    memoryprofiler_ptr_to_size[ptr] = size;
    if (!memoryprofiler_page_prerun_is_finished) { // Also track if this was a _malloc performed at preRun time.
      memoryprofiler_prerun_mallocs[ptr] = size;
    }

    if (MEMORYPROFILER_ALLOC_STATS) {
      var loc = new Error().stack.toString();//.split('\n');
//      for(var i in loc) {
        var str = loc;//[i];
        if (!memoryprofiler_alloc_site_stats[str]) {
          memoryprofiler_alloc_site_stats[str] = 1;
        } else {
          memoryprofiler_alloc_site_stats[str]++;
        }
 //     }
      memoryprofiler_alloc_site_ptrs[ptr] = loc;
    }
    // If this is a large enough allocation, track its detailed callstack info.
    if (size > MEMORYPROFILER_TRACK_CALLSTACK_MIN_SIZE) {
      // A very very hacky way to get the caller function as string. TODO: Once emscripten_get_callstack_js lands, use that instead.
      var loc = new Error().stack.toString();
      var nl = loc.indexOf('\n')+1;
      loc = loc.substr(nl);
      loc = loc.replace(/\n/g, '<br />');
//      nl = loc.indexOf('\n', nl)+1;
      /*
      if (nl != -1 && loc[nl] != '_') {
        nl = loc.indexOf('\n', nl)+1;
      }
      if (nl != -1 && loc.substr(nl, 4) == '__Zn') {
        nl = loc.indexOf('\n', nl)+1;
      }
      var nl2 = loc.indexOf('\n', nl);*/
      var caller = loc;//.substr(nl, nl2-nl);
      memoryprofiler_ptr_to_loc[ptr] = caller;
      if (memoryprofiler_loc_to_size[caller] > 0)
        memoryprofiler_loc_to_size[caller] += size;
      else
        memoryprofiler_loc_to_size[caller] = size;
    }
    return ptr;
  }

  var prevFree = _free;
  function hookedFree(ptr) {
    if (ptr) {
      // Decrement global stats.
      var sz = memoryprofiler_ptr_to_size[ptr];
      memoryprofiler_total_mem_allocated -= sz;
      delete memoryprofiler_ptr_to_size[ptr];
      delete memoryprofiler_prerun_mallocs[ptr]; // Also free if this happened to be a _malloc performed at preRun time.
      memoryprofiler_stacktop_watermark = Math.max(memoryprofiler_stacktop_watermark, STACKTOP);

      if (MEMORYPROFILER_ALLOC_STATS) {
        var loc = memoryprofiler_alloc_site_ptrs[ptr];
        if (loc) {
//          for(var i in loc) {
            var str = loc;//[i];
            memoryprofiler_alloc_site_stats[str]--;
//          }
        }
        memoryprofiler_alloc_site_ptrs[ptr] = null;
      }

      // Decrement per-alloc stats if this was a large allocation.
      if (sz > MEMORYPROFILER_TRACK_CALLSTACK_MIN_SIZE) {
        var caller = memoryprofiler_ptr_to_loc[ptr];
        delete memoryprofiler_ptr_to_loc[ptr];
        memoryprofiler_loc_to_size[caller] -= sz;
        if (memoryprofiler_loc_to_size[caller] <= 0) {
          delete memoryprofiler_loc_to_size[caller];
        }
      }
    }
    ++memoryprofiler_num_frees;
    return prevFree(ptr);
  }
  // Inject the memoryprofiler malloc() and free() hooks.
  _malloc = hookedMalloc;
  _free = hookedFree;
  // Also inject the same pointers in the Module object.
  Module['_malloc'] = hookedMalloc;
  Module['_free'] = hookedFree;

  // Add a tracking mechanism to detect when VFS loading is complete.
  function detectPreloadComplete() {
    memoryprofiler_page_prerun_is_finished = true;
  }
  Module['preRun'].push(detectPreloadComplete); // This is will be the last preRun task to be run. Assuming that nobody will add new tasks to preRun after this.
  // BUG! Looks like if there is no filesystem to preload, the above detectPreloadComplete() handler will not get run! Therefore hook into the postRun event as well!
  // This will get run, although in that case we will mistakenly track some allocations as if they were preRun allocs, even though they aren't. TODO: Come up with a proper fix.
  Module['postRun'].push(detectPreloadComplete); // This is will be the last preRun task to be run. Assuming that nobody will add new tasks to preRun after this.

  if (MEMORYPROFILER_HOOK_STACKALLOC) {
    // Inject stack allocator.
    var prevStackAlloc = Runtime.stackAlloc;
    function hookedStackAlloc(size) {
      memoryprofiler_stacktop_watermark = Math.max(memoryprofiler_stacktop_watermark, STACKTOP + size);
      return prevStackAlloc(size);
    }
    Runtime.stackAlloc = hookedStackAlloc;
  }

  memoryprofiler = document.getElementById('memoryprofiler');
  if (!memoryprofiler) {
    var div = document.createElement("div");
    div.innerHTML = "<div style='border: 2px solid black; padding: 2px;'><canvas style='border: 1px solid black;' id='memoryprofiler_canvas' width='800' height='50'></canvas>MEMORYPROFILER_TRACK_CALLSTACK_MIN_SIZE=<input id='memoryprofiler_min_tracked_alloc_size' type=number onChange='MEMORYPROFILER_TRACK_CALLSTACK_MIN_SIZE=this.value;' value="+MEMORYPROFILER_TRACK_CALLSTACK_MIN_SIZE+"></input><br/><input type='checkbox' onchange='MEMORYPROFILER_ALLOC_STATS=this.checked;'>Print allocation statistics by callstack to html log (warning: slow!)</input><input type='button' value='Clear alloc stats' onclick='memoryprofiler_alloc_site_stats = {}; memoryprofiler_alloc_site_ptrs = {};'></input><div id='memoryprofiler'></div>";
    document.body.appendChild(div);
    memoryprofiler = document.getElementById('memoryprofiler');
  }
  
  memoryprofiler_canvas = document.getElementById('memoryprofiler_canvas');
  memoryprofiler_canvas.width = document.documentElement.clientWidth - 64;
  memoryprofiler_canvas_size = memoryprofiler_canvas.width * memoryprofiler_canvas.height;
  memoryprofiler_canvas_context = memoryprofiler_canvas.getContext('2d');

  memoryprofiler_update_ui();
  setInterval(memoryprofiler_update_ui, MEMORYPROFILER_UI_UPDATE_INTERVAL);
}

// Given a pointer 'bytes', compute the linear 1D position on the graph as pixels, rounding down for start address of a block.
function memoryprofiler_bytesToPixels_rounddown(bytes) {
  return (bytes * memoryprofiler_canvas_size / TOTAL_MEMORY) | 0;
}

// Same as memoryprofiler_bytesToPixels_rounddown, but rounds up for the end address of a block. The different rounding will
// guarantee that even 'thin' allocations should get at least one pixel dot in the graph.
function memoryprofiler_bytesToPixels_roundup(bytes) {
  return ((bytes * memoryprofiler_canvas_size + TOTAL_MEMORY - 1) / TOTAL_MEMORY) | 0;
}

// Graphs a range of allocated memory. The memory range will be drawn as a top-to-bottom, left-to-right stripes or columns of pixels.
function memoryprofiler_fillLine(startBytes, endBytes) {
  var startPixels = memoryprofiler_bytesToPixels_rounddown(startBytes);
  var endPixels = memoryprofiler_bytesToPixels_roundup(endBytes);

  // Starting pos (top-left corner) of this allocation on the graph.
  var x0 = (startPixels / memoryprofiler_canvas.height) | 0;
  var y0 = startPixels - x0 * memoryprofiler_canvas.height;
  // Ending pos (bottom-right corner) of this allocation on the graph.
  var x1 = (endPixels / memoryprofiler_canvas.height) | 0;
  var y1 = endPixels - x1 * memoryprofiler_canvas.height;
  
  // Draw the left side partial column of the allocation block.
  if (y0 > 0 && x0 < x1) {
    memoryprofiler_canvas_context.fillRect(x0,y0,1,memoryprofiler_canvas.height-y0);
    // Proceed to the start of the next full column.
    y0 = 0;
    ++x0;
  }
  // Draw the right side partial column.
  if (y1 < memoryprofiler_canvas.height && x0 < x1) {
    memoryprofiler_canvas_context.fillRect(x1,0,1,y1);
    // Decrement to the previous full column.
    y1 = memoryprofiler_canvas.height-1;
    --x1;
  }
  // After filling the previous leftovers with one-pixel-wide lines, we are only left with a rectangular shape of full columns to blit.
  memoryprofiler_canvas_context.fillRect(x0,0,x1+1-x0,memoryprofiler_canvas.height);
}

function memoryprofiler_count_openal_audiodata_size() {
  if (typeof AL == "undefined" || !AL.currentContext) {
    return 0;
  }

  var totalMemory = 0;

  for(var i in AL.currentContext.buf) {
    var buffer = AL.currentContext.buf[i];
    for(var channel = 0; channel < buffer.numberOfChannels; ++channel) {
      totalMemory += buffer.getChannelData(channel).length * 4;
    }
  }
  return totalMemory;
}

// Main UI update entry point.
function memoryprofiler_update_ui() {

  function colorBar(color) {
//    return '<span style="padding:0px; border:solid 1px black; width:32px;height:18px; display:inline-block;"><span style="display:inline-block; width:30px; height: 16px; background-color:'+color+'; margin: 1px;"></span></span>';
    return '<span style="padding:0px; border:solid 1px black; width:28px;height:14px; vertical-align:middle; display:inline-block; background-color:'+color+';"></span>';
  }
  
  // Naive function to compute how many bits will be needed to represent the number 'n' in binary. This will be our pointer 'word width' in the UI.
  function nBits(n) {
    var i = 0;
    while(n >= 1) {
      ++i;
      n /= 2;
    }
    return i;
  }

  // Returns i formatted to string as fixed-width hexadecimal.
  function toHex(i, width) {
    var str = i.toString(16);
    while(str.length < width) {
      str = '0' + str;
    }
    return '0x'+str;
  }
  var width = (nBits(TOTAL_MEMORY)+3)/4; // Pointer 'word width'
  memoryprofiler.innerHTML = 'Total HEAP size: ' + formatBytes(TOTAL_MEMORY) + '.';
  memoryprofiler.innerHTML += '<br />'+colorBar('#202020')+'STATIC memory area size: ' + formatBytes(STATICTOP-STATIC_BASE);
  memoryprofiler.innerHTML += '. STATIC_BASE: ' + toHex(STATIC_BASE, width);
  memoryprofiler.innerHTML += '. STATICTOP: ' + toHex(STATICTOP, width) + '.';
  
  memoryprofiler.innerHTML += '<br />'+colorBar('#FF8080')+'STACK memory area size: ' + formatBytes(STACK_MAX-STACK_BASE);
  memoryprofiler.innerHTML += '. STACK_BASE: ' + toHex(STACK_BASE, width);
  memoryprofiler.innerHTML += '. STACKTOP: ' + toHex(STACKTOP, width);
  memoryprofiler.innerHTML += '. STACK_MAX: ' + toHex(STACK_MAX, width) + '.';
  memoryprofiler.innerHTML += '<br />STACK memory area used now (should be zero): ' + formatBytes(STACKTOP-STACK_BASE) + '.'+colorBar('#FFFF00')+' STACK watermark highest seen usage (approximate lower-bound!): ' + formatBytes(memoryprofiler_stacktop_watermark-STACK_BASE);
  
  memoryprofiler.innerHTML += '<br />'+colorBar('#70FF70')+'DYNAMIC memory area size: ' + formatBytes(DYNAMICTOP-DYNAMIC_BASE);
  memoryprofiler.innerHTML += '. DYNAMIC_BASE: ' + toHex(DYNAMIC_BASE, width);
  memoryprofiler.innerHTML += '. DYNAMICTOP: ' + toHex(DYNAMICTOP, width) + '.';
  memoryprofiler.innerHTML += '<br />'+colorBar('#6699CC')+colorBar('#003366')+colorBar('#0000FF')+'DYNAMIC memory area used: ' + formatBytes(memoryprofiler_total_mem_allocated) + ' (' + (memoryprofiler_total_mem_allocated*100.0/(TOTAL_MEMORY-DYNAMIC_BASE)).toFixed(2) + '% of all free memory)';
  
  var preloadedMemoryUsed = 0;
  for(i in memoryprofiler_prerun_mallocs) {
    preloadedMemoryUsed += memoryprofiler_prerun_mallocs[i]|0;
  }
  memoryprofiler.innerHTML += '<br />'+colorBar('#FF9900')+colorBar('#FFDD33')+'Preloaded memory used, most likely memory reserved by files in the virtual filesystem : ' + formatBytes(preloadedMemoryUsed);

  memoryprofiler.innerHTML += '<br />OpenAL audio data: ' + formatBytes(memoryprofiler_count_openal_audiodata_size()) + ' (outside HEAP)';
  memoryprofiler.innerHTML += '<br />'+colorBar('#FFFFFF')+'Unallocated HEAP space: ' + formatBytes(TOTAL_MEMORY - DYNAMICTOP);
  memoryprofiler.innerHTML += '<br /># of total malloc()s/free()s performed in app lifetime: ' + memoryprofiler_num_allocs + '/' + memoryprofiler_num_frees + ' (delta: ' + (memoryprofiler_num_allocs-memoryprofiler_num_frees) + ')';
  
  // Background clear
  memoryprofiler_canvas_context.fillStyle="#FFFFFF";
  memoryprofiler_canvas_context.fillRect(0, 0, memoryprofiler_canvas.width, memoryprofiler_canvas.height);

  memoryprofiler_canvas_context.fillStyle="#202020";
  memoryprofiler_fillLine(STATIC_BASE, STATICTOP);

  memoryprofiler_canvas_context.fillStyle="#FF8080";
  memoryprofiler_fillLine(STACK_BASE, STACK_MAX);

  memoryprofiler_canvas_context.fillStyle="#FFFF00";
  memoryprofiler_fillLine(STACK_BASE, memoryprofiler_stacktop_watermark);

  memoryprofiler_canvas_context.fillStyle="#FF0000";
  memoryprofiler_fillLine(STACK_BASE, STACKTOP);
  
  memoryprofiler_canvas_context.fillStyle="#70FF70";
  memoryprofiler_fillLine(DYNAMIC_BASE, DYNAMICTOP);
  
  if (MEMORYPROFILER_DETAILED_HEAP_USAGE) {
    // Print accurate map of individual allocations. This will show information about 
    // memory fragmentation and allocation sizes.
    // Warning: This will walk through all allocations, so it is slow!
    function printAllocsWithCyclingColors(colors, allocs) {
      var colorIndex = 0;
      for(var i in allocs) {
        memoryprofiler_canvas_context.fillStyle=colors[colorIndex];
        colorIndex = (colorIndex+1)%colors.length;
        var start = i|0;
        var sz = allocs[start]|0;
        memoryprofiler_fillLine(start, start+sz);
      }
    }
    
    printAllocsWithCyclingColors([ "#6699CC", "#003366", "#0000FF" ], memoryprofiler_ptr_to_size);
    printAllocsWithCyclingColors([ "#FF9900", "#FFDD33" ], memoryprofiler_prerun_mallocs);

  } else {
    // Print only a single naive blob of individual allocations. This will not be accurate, but is constant-time.
    memoryprofiler_canvas_context.fillStyle="#0000FF";
    memoryprofiler_fillLine(DYNAMIC_BASE, DYNAMIC_BASE+memoryprofiler_total_mem_allocated);
  }
  
  function isEmpty(cont) {
    for(i in cont) {
      return false;
    }
    return true;
  }
  
  // Print out statistics of individual allocations if they were tracked.
   if (!isEmpty(memoryprofiler_loc_to_size)) {
    memoryprofiler.innerHTML += '<h4>Notable allocation sites<h4>'
    for(var i in memoryprofiler_loc_to_size) {
      memoryprofiler.innerHTML += '<b>'+formatBytes(memoryprofiler_loc_to_size[i]|0)+'</b>: ' + i + '<br />';
    }
  }

  if (!isEmpty(memoryprofiler_alloc_site_stats)) {
    var calls = [];
    for(var i in memoryprofiler_alloc_site_stats) {
      var numcalls = memoryprofiler_alloc_site_stats[i];
      if (numcalls >= MEMORYPROFILER_ALLOC_STATS_MIN_REPORTED) {
        calls.push(i);
      }
    }

    calls.sort(function(a,b) { return memoryprofiler_alloc_site_stats[b] - memoryprofiler_alloc_site_stats[a]; });
    memoryprofiler.innerHTML += '<h4>Allocated pointers by call stack:<h4>';
    var ndemangled = 10;
    for(var i in calls) {
      var callstack = calls[i];
      var numcalls = memoryprofiler_alloc_site_stats[callstack];
      if (ndemangled > 0) {
        callstack = demangleAll(callstack);
        callstack = callstack.split('\n').join('<br />');
        --ndemangled;
      }
      memoryprofiler.innerHTML += callstack + ': <b>' + numcalls + '</b><br /><br />';
    }
  }

  // Reset watermark for until next UI update.
//  memoryprofiler_stacktop_watermark = STACK_BASE;
}
