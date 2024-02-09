// static/js/main.js
$(document).ready(function(){
    $("form").on("submit", function(event){
        event.preventDefault();
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: $(this).serialize(),
            success: function(data){
                var resultParagraph = $("#result");
                resultParagraph.text("This is news a " + data.label+" News");
                resultParagraph.removeClass("real fake");
                resultParagraph.addClass(data.label.toLowerCase());
            }
        });
    });
});
$("#clearButton").click(function(){
    $("form")[0].reset();
    $("#result").text('');
});