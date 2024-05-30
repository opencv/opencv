function getLabelName(innerHTML) {
    var str = innerHTML.toLowerCase();
    // Replace all '+' with 'p'
    str = str.split('+').join('p');
    // Replace all ' ' with '_'
    str = str.split(' ').join('_');
    // Replace all '#' with 'sharp'
    str = str.split('#').join('sharp');
    // Replace other special characters with 'ascii' + code
    for (var i = 0; i < str.length; i++) {
        var charCode = str.charCodeAt(i);
        if (!(charCode == 95 || (charCode > 96 && charCode < 123) || (charCode > 47 && charCode < 58)))
            str = str.substr(0, i) + 'ascii' + charCode + str.substr(i + 1);
    }
    return str;
}

function addToggle() {
    var $getDiv = $('div.newInnerHTML').last();
    var buttonName = $getDiv.html();
    var label = getLabelName(buttonName.trim());
    $getDiv.attr("title", label);
    $getDiv.hide();
    $getDiv = $getDiv.next();
    $getDiv.attr("class", "toggleable_div label_" + label);
    $getDiv.hide();
}

function addButton(label, buttonName) {
    var b = document.createElement("BUTTON");
    b.innerHTML = buttonName;
    b.setAttribute('class', 'toggleable_button label_' + label);
    b.onclick = function() {
        $('.toggleable_button').css({
            border: '2px outset',
            'border-radius': '4px'
        });
        $('.toggleable_button.label_' + label).css({
            border: '2px inset',
            'border-radius': '4px'
        });
        $('.toggleable_div').css('display', 'none');
        $('.toggleable_div.label_' + label).css('display', 'block');
    };
    b.style.border = '2px outset';
    b.style.borderRadius = '4px';
    b.style.margin = '2px';
    return b;
}

function buttonsToAdd($elements, $heading, $type) {
    if ($elements.length === 0) {
        return;
    }
    var arr = jQuery.makeArray($elements);
    var seen = {};
    arr.forEach(function(e) {
        var txt = e.innerHTML;
        if (!seen[txt]) {
            $button = addButton(e.title, txt);
            if (Object.keys(seen).length == 0) {
                var linebreak1 = document.createElement("br");
                var linebreak2 = document.createElement("br");
                ($heading).append(linebreak1);
                ($heading).append(linebreak2);
            }
            ($heading).append($button);
            seen[txt] = true;
        }
    });
    return;
}

function addTutorialsButtons() {
    $("h1").each(function() {
        var $elements = $(this).nextUntil("h1")
        var $lower = $elements.find("div.newInnerHTML")
        $elements = $elements.add($lower)
        $elements = $elements.filter("div.newInnerHTML")
        buttonsToAdd($elements, $(this), "h1")
    });
    $(".toggleable_button").first().click();
    var $clickDefault = $('.toggleable_button.label_python').first();
    if ($clickDefault.length) {
        $clickDefault.click();
    }
    $clickDefault = $('.toggleable_button.label_cpp').first();
    if ($clickDefault.length) {
        $clickDefault.click();
    }
    return;
}
