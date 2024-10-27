window.onscroll = function() {
    var body = document.body;
    var scrollTop = document.documentElement.scrollTop || body.scrollTop;

    if (scrollTop > 50) {  // Trigger the shrinking effect when scrolling past 50px
        body.classList.add("scrolled");
    } else {
        body.classList.remove("scrolled");
    }
};

