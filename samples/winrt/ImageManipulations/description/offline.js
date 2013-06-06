var Galleries = Galleries || { };

(function() {

    function findElem(parent, tagName, className) {
        var elemToSearch = (parent) ?  parent : document.body;
        var tagMatch = elemToSearch.getElementsByTagName(tagName);
        var evaluator = function(elem) {
            return (className) ? (elem.className.indexOf(className) > -1) : true;
        };

        return findArrayElem(tagMatch, evaluator);
    }

    function findArrayElem(array, evaluator) {
        var newArray = new Array();
        for (var count = 0; count < array.length; count++) {
            if (evaluator(array[count])) {
                newArray.push(array[count]);
            }
        }
        return newArray;
    }

    function iterateElem(elems, delegate) {
        for(var count = 0; count < elems.length; count++) {
            delegate(count, elems[count]);
        }
    }

    function isHidden(elem) {
        return (elem.offsetHeight === 0 && elem.offsetWidth === 0) || elem.style && elem.style.display === "none";
    }
    
    function onWindowLoad(callback) {
        attachEventHandler(null, 'load', callback);
    }
 
    function attachEventHandler(elem, event, callback) {
        var elemToAttach = (elem) ? elem : window;
        if (document.addEventListener) {
			elemToAttach.addEventListener(event, callback, false);
		} else if ( document.attachEvent ) {
			elemToAttach.attachEvent('on' + event, callback);
		}
    }

    Galleries.findElem = findElem;
    Galleries.iterateElem = iterateElem;
    Galleries.attachEventHandler = attachEventHandler;
    Galleries.onWindowLoad = onWindowLoad;
})();