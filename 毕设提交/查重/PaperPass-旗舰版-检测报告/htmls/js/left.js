$(function () {
	//  传值给detail_report
	parent.postMessage("width_70", "*");
	// 传值给main
	parent.parent.postMessage("word", "*");
	// 页码加载
	var str = "";
	str += `<span class="pageNum"></span><span class="partition"></span><span class="totalPage"></span>`;
	$(".top_Num").html(str);
	$(".pageNum").text(1);
	$(".partition").text("/");
	$(".top_Num").css({
		"background": "#000000",
		"height": "30px",
		"line-height": "30px",
		"padding": "0 10px",
	});
	$(".top_Num").show();
	// 切换模式按钮
	$(".Switch").click(function () {
		$(".layui-layer-shade").css("display", "block");
		$(".layui-layer").css("display", "block");
		$(".layui-layer-title").html("切换模式");
		$(".layui-layer-content").html("确认切换至“纯文字模式”吗？");
		$(".detail_btn").data('type', 'switch');
	});

	$(".detail_btn").click(function () {
		type = $(this).data('type');
		if (type == 'switch') {
			$(".layui-layer-btn0").attr("href", $(this).data('txtLink'));
			parent.postMessage("word_right", "*");
		};
	});
	$(".layui-layer-setwin").click(function () {
		$(".layui-layer-shade").css("display", "none");
		$(".layui-layer").css("display", "none");
	});
	$(".layui-layer-btn1,.layui-layer-btn0 ").click(function () {
		$(".layui-layer-shade").css("display", "none");
		$(".layui-layer").css("display", "none");
	});
	// 辅助线按钮
	$(".subline").click(function () {
		$(".layui-layer-shade").css("display", "block");
		$(".layui-layer").css("display", "block");
		$(".layui-layer-title").html("辅助线");
		curHref = $(".detail_btn").attr("href");
		if (curHref != '#') {
			$(".detail_btn").data('txtLink', $(".detail_btn").attr("href"));
			$(".layui-layer-btn0").attr("href", "#");
		};
		$(".detail_btn").data('type', 'subline');
		if ($(".red").parent().eq(0).hasClass('red_a')) {
			$(".layui-layer-content").html("确认“取消辅助线”吗？");
		} else {
			$(".layui-layer-content").html("确认“使用辅助线”吗？");
		};
	});
	function FuncLine(Line) {
		if (Line == "true") {
			$(".red").parent().removeClass('red_a');
			$(".orange").parent().removeClass('orange_a');
			$(".subline").css("background-position", "-106px -162px");
		} else {
			$(".red").parent().addClass('red_a');
			$(".orange").parent().addClass('orange_a');
			$(".subline").css("background-position", "-146px -159px");
		};
	};
	$('body').on('click', '.layui-layer-btn0', function () {
		if ($(".layui-layer-title").html() !== '辅助线') {
			return;
		};
		var has_Class = $(".red").parent().eq(0).hasClass('red_a');
		var str_has = String(has_Class);
		FuncLine(str_has);
	});
	// 滚动分页分页
	var totalPage = $(".upkmzjakmc").length;
	$('.totalPage').text(totalPage);
	// 滚动事件
	var nowPage = 1;

	function scrollFunc(size) {
		if (size === undefined) {
			var offsetH = $("body").scrollTop();
			var scrollH = 0;
			$('.upkmzjakmc').each(function (k, v) {
				var PageH = $(v)[0].getBoundingClientRect().height + 15;
				if (scrollH < offsetH && offsetH <= (scrollH + PageH)) {
					nowPage = k + 1;
				};
				scrollH += PageH;
			});
			$('.input_pageNum').val(nowPage);
			$('.pageNum').text(nowPage);
			$("#toall_Num").attr("href", "#" + nowPage);
		} else {
			var offsetH = $("body").scrollTop();
			var scrollH = 0;
			$('.upkmzjakmc').each(function (k, v) {
				var PageH = ($(v)[0].getBoundingClientRect().height) * size + 15;
				if (scrollH <= offsetH && offsetH <= (scrollH + PageH)) {
					nowPage = k + 1;
				};
				scrollH += PageH;
			});
			$('.input_pageNum').val(nowPage);
			$('.pageNum').text(nowPage);
			$("#toall_Num").attr("href", "#" + nowPage);
		};
	};
	$("body").scroll(function (e) {
		scrollFunc();
	});

	// 上下页
	$(".bottom").click(function () {
		$(".bottom_a").attr("href", "#" + (nowPage + 1));
		$("#toall_Num").attr("href", "#" + (nowPage + 1));
	});
	$(".top").click(function () {
		$(".top_a").attr("href", "#" + (nowPage - 1));
		$("#toall_Num").attr("href", "#" + (nowPage - 1));
	});
	// input获取，失去焦点
	$(".input_pageNum").focus(function () {
		$(".input_pageNum").css("background-color", "#424649");
		$(".input_pageNum").select();
	}).blur(function () {
		$(".input_pageNum").css("background-color", "transparent");
		page1 = $(".input_pageNum").val();
		if (page1 <= totalPage && /^[1-9]\d*$/.test(page1)) {
			$("#toall_Num").attr("href", "#" + page1);
		} else {
			$(".input_pageNum").val(nowPage);
		};
	}).mousemove(function () { // 鼠标移入移出
		$(".input_pageNum").css("background-color", "#424649");
	}).mouseout(function () {
		if ($('.input_pageNum').is(':focus')) {
			$(".input_pageNum").css("background-color", "#424649");
		} else {
			$(".input_pageNum").css("background-color", "transparent");
		};
	}).keyup(function () {
		if (event.keyCode == 13) {
			page1 = $(".input_pageNum").val();
			if (page1 <= totalPage && /^[1-9]\d*$/.test(page1)) {
				$("#toall_Num").attr("href", "#" + page1);
				window.location.hash = "#" + page1;
			} else {
				$(".input_pageNum").val(nowPage);
				$(".input_pageNum").blur();
			};
		};
	});
	// 鼠标移入移出改变背景
	$(".toall_div").mousemove(function () {
		$(this).css({
			"background-color": "#424649",
			"border-radius": "2px"
		});
		$(this).children(".tip").css("display", "block");
	}).mouseout(function () {
		$(this).css("background-color", "transparent");
		$(this).children(".tip").css("display", "none");
	});
	// 设置stl_02外层div的宽
	var arr = new Array();
	$('.upkmzjakmc').each(function (k, v) {
		var pageW = $(this)[0].getBoundingClientRect().width + 30;
		$(this).parent(".jcjymywwzd").css("width", pageW);
	});
	// 计算left和高度
	var w1 = screen.width;
	var h1 = screen.height;
	var w = document.documentElement.clientWidth;
	var h = document.documentElement.clientHeight;
	var left = (w - minWidth) / 2;
	var leftWidth = left - 75;
	var rightWidth = left + 15;
	var shadeLeft = (w - 500) / 2;
	var shadeTop = (h1 - 273) / 2;
	$(".Shadow ").css("width", w - 17);
	$(".top_Num").css("right", rightWidth);
	$("#body").height(h);
	$(".overflow_ul").css("max-height", h - 100);
	$(".layui-nav").css("max-width", w * 0.29);
	$(".layui-layer").css({
		"left": shadeLeft,
		"top": "60px"
	});
});
var arr = new Array();
$('.upkmzjakmc').each(function (k, v) {
	arr.push($(this).width());
});
function getMaxCountWidth(arr){
	var mp=new Array();
	for(var idx in arr){
		var item=arr[idx];
		var count=0;
		if(!mp.hasOwnProperty(item)){
			mp[item]=1;
		}else{
			count=mp[item];
			count=count+1;
			mp[item]=count;
		};
	};
	var maxCount=0;
	var width=0;
	for(var key in mp){
		var val=mp[key];
		if(val>=maxCount){
			maxCount=val;
			if(width==0){
				width=key;
			}else if(key<width){
				width=key;
			};
		};
	};
return width;
};
var minWidth=getMaxCountWidth(arr);
// 拖动窗口的高度
$(window).resize(function () {
	var w1 = screen.width;
	var w = document.documentElement.clientWidth;
	var h = document.documentElement.clientHeight;
	var left = (w - minWidth) / 2;
	var leftWidth = left - 15;
	var rightWidth = left + 15;
	var shadeLeft = (w - 500) / 2;
	var shadeTop = (h - 273) / 2;
	$(".top_Num").css("right", rightWidth);
	$(".Shadow ").css("width", w - 17);
	$("#body").height(h);
	$(".overflow_ul").css("max-height", h - 100);
	$(".layui-nav").css("max-width", w * 0.29);
	$(".layui-layer").css({
		"left": shadeLeft,
		"top": "60px"
	});
});
