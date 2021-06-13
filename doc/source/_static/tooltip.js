function getTopbarHeight() {
    var divs = document.getElementsByTagName('div');
    for (var i = 0; i < divs.length; i++){

        // check if the class name contains the substring 'topbar'
        // taken from https://attacomsian.com/blog/ways-to-check-string-contains-substring-javascript
        if (divs[i].className.indexOf('topbar') >= 0){

            // get height of the <div> element
            // inspired by https://stackoverflow.com/a/10465984
            return(parseFloat(window.getComputedStyle(divs[i]).height));
        }
    }
}

// taken from https://stackoverflow.com/a/14446538
function readTextFile(file)
{
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = function ()
    {
        if(rawFile.readyState === 4)
        {
            if(rawFile.status === 200 || rawFile.status == 0)
            {
                var allText = rawFile.responseText;
                alert(allText);
                return(allText);
            }
        }
    }
    rawFile.send(null);
}

// based on http://michaelsoriano.com/better-tooltips-with-plain-javascript-css/
//
// additional links and resources:
// - different options for running JavaScript when page is loading
//   - https://stackoverflow.com/a/2920207
// - how to pass attributes to the '<script>' tag from sphinx
//   - https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_js_files
// - 'Cross-Origin Request Blocked' when adding script with 'type="module"'
//   - https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS/Errors/CORSRequestNotHttp
//   - https://support.mozilla.org/en-US/questions/1264280 
//   - in Firefox about:config change 'privacy.file_unique_origin' to 'false'

var abbrs = document.getElementsByTagName('abbr');
// window.alert(abbrs.length);
for (var i = 0; i < abbrs.length; i++){
    abbrs[i].addEventListener('mouseover', createCustomTooltip);
    abbrs[i].addEventListener('mouseout', cancelCustomTooltip);
}

var terms = document.getElementsByClassName('xref std std-term');
for (var i = 0; i < terms.length; i++){
    terms[i].addEventListener('mouseover', createGlossaryTooltip);
    terms[i].addEventListener('mouseout', cancelGlossaryTooltip);
}

// https://stackoverflow.com/a/24378510
glossary = JSON.parse(glossary);


function createCustomTooltip(ev){
    console.log('createTip');
    // remove the standard tooltip (stored in the attribute 'title')
    // and temporarily store it in a new custom attribute 'tooltip'
    // that is not interpreted by the browser
    var title = this.title;
    this.title = '';
    this.setAttribute('tooltip', title);

    var tooltipWrap = document.createElement('div');
    tooltipWrap.className = 'custom-tooltip'; // adds class
    tooltipWrap.appendChild(document.createTextNode(title)); // add text node to the new div
    console.log(tooltipWrap);

    var parent = this.parentNode;
    parent.insertBefore(tooltipWrap, this);

    var padding = 5;
    var linkProps = this.getBoundingClientRect();
    var tooltipProps = tooltipWrap.getBoundingClientRect();

    // https://www.gavsblog.com/blog/get-the-current-position-of-the-mouse-from-a-javascript-event
    // var mouseX = ev.clientX;
    // var mouseY = ev.clientY;
    // var topPos = mouseY - (tooltipProps.height + padding);
    // var leftPos = mouseX - 0.5*tooltipProps.width;
    var topPos = linkProps.top + 0.5*linkProps.height - 0.5*tooltipProps.height;
    var leftPos = linkProps.right + padding;
    tooltipWrap.setAttribute('style', 'top:'+topPos+'px; left:'+leftPos+'px;');
}

function cancelCustomTooltip(ev){
    // restore the standard tooltip by retrieving the tooltip text
    // from the custom attribute 'tooltip' and restoring it to the
    // 'title' attribute which normally holds the tooltip text
    // finally clean up by deleting the temporary attribute 'tooltip'
    var title = this.getAttribute('tooltip');
    this.title = title;
    this.removeAttribute('tooltip');

    // remove all tooltips
    document.querySelector(".custom-tooltip").remove();
}

function createGlossaryTooltip(ev){
    // figure out the key/name of the glossary entry
    console.log('this.innerHTML = ' + this.innerHTML)

    // display the tooltip
    var tooltipWrap = document.createElement('div');
    tooltipWrap.className = 'glossary-tooltip';
    var content = glossary[this.innerHTML.toLowerCase()].replaceAll("&quot;", '"'); // https://stackoverflow.com/a/1145525
    tooltipWrap.innerHTML = `
<div class="glossary-tooltip-header">
    ` + this.innerHTML + `
</div>
<div class="glossary-tooltip-body">
    ` + content + `
</div>`;

    var parent = this.parentNode;
    parent.insertBefore(tooltipWrap, this);

    // trigger MathJax rendering
    // taken from https://stackoverflow.com/a/36225278
    MathJax.Hub.Queue(['Typeset', MathJax.Hub, tooltipWrap]);

    // positions and sizes of the glossary link and the glossary tooltip
    var linkProps = this.getBoundingClientRect();
    var tooltipProps = tooltipWrap.getBoundingClientRect();

    // calculate correct glossary tooltip position
    // - https://codingfortech.com/how-to-javascript-get-element-width-by-examples/
    // - https://www.kirupa.com/html5/get_element_position_using_javascript.htm
    const distance = 2; // distance of glossary tooltip from glossary link/term
    const margin = 10; // margin between the glossary tooltip and the window borders
    const yMin = getTopbarHeight() + margin;
    const yMax = window.innerHeight - margin;
    const yDefault = linkProps.top + 0.5*linkProps.height - 0.5*tooltipProps.height;
    var topPos = Math.max(yMin, Math.min(yDefault, yMax-tooltipProps.height));
    // var leftPos = linkProps.right + distance;
    var nav = document.getElementById('site-navigation') // nav-bar on the left
    var leftPos = linkProps.left - distance - tooltipProps.width;
    if(leftPos < nav.offsetWidth + margin){
        leftPos = linkProps.right + distance;
    }
    tooltipWrap.setAttribute('style', 'top:'+topPos+'px; left:'+leftPos+'px;');

}

function cancelGlossaryTooltip(ev){
    document.querySelector(".glossary-tooltip").remove();
}
