(function(System,$){
$(function () {
	// 初始化目录树
	var treeData = JSON.parse(detail_json);
	tree = loop(treeData);
	$('#detail_report').html(tree);
	$('#detail_report').children('.layui-nav-third-child').removeClass('layui-bg-white').removeClass(
		'layui-nav-third-child');
	var originalData = JSON.parse(original_json);
	original = loopFunc(originalData);
	$("#original_report").html(original);
	$('#original_report').children('.layui-nav-third-child').removeClass('layui-bg-white').removeClass(
		'layui-nav-third-child');
	// 递归加载树
	function loop(dt) {
		var str = "";
		if (dt.length == 0) {
			$(".mclick_show").hide();
			str += '<div style="color:#000000">' + "目录为空" + '</div>';
		} else {
			dt.forEach(item => {
				var children = (typeof (item.children) != "undefined") && item.children && item.children
					.length ? loop(item.children) : null;
				if (children != null) {
					//当前节点+子节点
					str +=
						'<div class="layui-nav-third-child layui-bg-white">' +
						'<div class="mselect_left">' +
						'<div class="mselect_img select_img_detail"></div>' +
						'<div class="mselect_tit">' +
						'<a href="htmls/word/word_report.html' + item.anchorPoint +
						'" target="left" class="third-class">' + item.content + '</a>' +
						'</div>' +
						'</div>' + children +
						'</div>';
				} else {
					//当前节点
					str +=
						'<div class="layui-nav-third-child layui-bg-white">' +
						'<div class="mselect_left ">' +
						'<div class="mselect_tit leftImg_none">' +
						'<a href="htmls/word/word_report.html' + item.anchorPoint +
						'" target="left" class="third-class">' + item.content + '</a>' +
						'</div>' +
						'</div>' +
						'</div>';
				};
			});
		};
		return str;
	};

	function loopFunc(dt) {
		var str = "";
		if (dt.length == 0) {
			$(".mclick_show").hide();
			str += '<div style="color:#000000">' + "目录为空" + '</div>';
		} else {
			dt.forEach(item => {
				var children = (typeof (item.children) != "undefined") && item.children && item.children
					.length ? loopFunc(item.children) : null;
				if (children != null) {
					//当前节点+子节点
					str += '<div class="layui-nav-third-child layui-bg-white">' +
						'<div class="mselect_left">' +
						'<div class="mselect_img mselect_img_ori"></div>' +
						'<div class="mselect_tit">' +
						'<a href="htmls/word/word_original.html' + item.anchorPoint +
						'" target="main" class="third-class">' + item.content + '</a>' +
						'</div>' +
						'</div>' + children +
						'</div>';
				} else {
					//当前节点
					str += '<div class="layui-nav-third-child layui-bg-white">' +
						'<div class="mselect_left ">' +
						'<div class="mselect_tit leftImg_none">' +
						'<a href="htmls/word/word_original.html' + item.anchorPoint +
						'" target="main" class="third-class">' + item.content + '</a>' +
						'</div>' +
						'</div>' +
						'</div>';
				};
			});
		};
		return str;
	};
	// Header收展按钮
	var html = $(".report_number").html();
	function FuncHeader(Header) {
		var h = document.body.clientHeight;
		var str = "";
		if (Header == 'block') {
			$('.paper-header').css('display', 'none');
			$(".clicktxt").text("展开");
			$(".clickimg").css("background-position", "-44px -27px");
			$("#m-content").height(h - 40);
			$("#childIframe").height(h - 40);
			$(".layui-nav").css("top", "55px");
			$(".overflow_ul").css("max-height", h - 140);
			$(".nav").css("padding", "0 40px 0 22px");
			$(".report_number").css({ "display": "flex", "justify-content": "flex-start", "justify-items": "center", "padding-left": "140px" });
			$(".tab_img").attr("src", "htmls/images/Header_logo.png");
			var similarNum = $(".similarNum").html();
			var progress=$(".nav-detail-similar .progress").html();
			str += `<div style="color:#e6e6e6">总体相似度：</div>
					<div style="width:80px;height:40px;line-height:40px;margin-right:10px;padding-top:4px">
						<div class="progress" style="margin-top:-5px">${progress}</div>	
					</div>
		            <div style="color:#e6e6e6;display:inline-block">${similarNum}</div>`;
		} else {
			$('.paper-header').css('display', 'block');
			$(".clicktxt").text("收起");
			$(".clickimg").css("background-position", "-78px -27px");
			$("#m-content").height(h - 140);
			$("#childIframe").height(h - 140);
			$(".layui-nav").css("top", "155px");
			$(".overflow_ul").css("max-height", h - 240);
			$(".nav").css("padding", "0 40px");
			$(".report_number").css("padding-left", "0px");
			$(".tab_img").attr("src", "");
			str = html;
		};
		$(".report_number").html(str);
	};
	var storage = window.localStorage;
	var HeaderState = System.operateState.getHeaderState(Report.report_id);
	FuncHeader(HeaderState);
	$(".HeaderButton").click(function () {
		var show = $('.paper-header').css('display');
		System.operateState.saveHeaderState(Report.report_id, show);
		FuncHeader(show);
	});
	// 截取目录字符串
	$('.third-class').each(function () {
		var max = 40;
		var str = $(this).text().trim();
		var length = str.length;
		if (length > max) {
			$(this).html(str.substring(0, max) + '...');
		};
	});
	// 点击nav隐藏目录
	$("#m-nav a").click(function () {
		$(".mlayui-nav-child").hide();
		$(".bookmark1_layui-nav-child").hide();
		$(".mclick_span").text("全部展开");
		$("i.g-arrow-i").css({
			"border-color": "#56b282 transparent transparent",
			"top": "18px"
		});
	});
	//存取tab a链接href
	$(".active_li").click(function () {
		var a_Attr = $(this).children("a").attr("href");
		var tab_name = $(this).children("a").text();
		System.operateState.saveALink(Report.report_id, a_Attr);
		System.operateState.saveTabName(Report.report_id, tab_name);
	});
	var aLink = System.operateState.getALink(Report.report_id);
	var tabName = System.operateState.getTabName(Report.report_id);
	$("#childIframe").attr("src", aLink);
	$(".active_li").each(function () {
		if ($(this).children("a").text() == tabName) {
			$(this).addClass('active');
			$(this).siblings().removeClass('active');
		};
	});
	// 目录关闭按钮
	$(".m_close").click(function () {
		$(".mlayui-nav-child").hide();
	});
	// 展开目录
	$(".mselect_img").on('click', function () {
		if ($(this).hasClass('opened')) {
			$(this).parent().siblings(".layui-nav-third-child").hide();
			$(this).parent().siblings(".layui-nav-third-child").find(".mselect_img").removeClass('opened');
			$(this).parent().siblings(".layui-nav-third-child").find(".mselect_img").css('background-position', '-115px -245px');
			$(this).css('background-position', '-115px -245px');
			$(this).removeClass('opened');
		} else {
			$(this).parent().siblings(".layui-nav-third-child").show();
			$(this).css('background-position', '-77px -245px');
			$(this).addClass('opened');
		};
	});
	// 详情报告页面按钮
	$(".select_img_detail").click(function () {
		var flag = true;
		$(".select_img_detail").each(function (k, v) {
			if (!$(v).hasClass('opened')) {
				flag = false;
			};
		});
		if (flag) {
			$(".mclick_span").text("全部收起");
			$("i.g-arrow-i").css({
				"border-color": "transparent transparent #56b282",
				"top": "13px"
			})
		} else {
			$(".mclick_span").text("全部展开");
			$("i.g-arrow-i").css({
				"border-color": "#56b282 transparent transparent",
				"top": "18px"
			});
		};
	});
	// 查看原文页面按钮
	$(".mselect_img_ori").click(function () {
		var flag = true;
		var mselect_img = $(".mselect_img_ori");
		$.each(mselect_img, function (k, v) {
			if (!$(v).hasClass('opened')) {
				flag = false;
			};
		});
		if (flag) {
			$(".mclick_span").text("全部收起");
			$("i.g-arrow-i").css({
				"border-color": "transparent transparent #56b282",
				"top": "13px"
			});
		} else {
			$(".mclick_span").text("全部展开");
			$("i.g-arrow-i").css({
				"border-color": "#56b282 transparent transparent",
				"top": "18px"
			});
		};
	});
	if($(".layui-nav-third-child").length==0){
		$(".mclick_show").hide();
	};
	// 全部展开收起
	$(".mclick_show").click(function () {
		var thirdchild = $(this).parent().parent().next().find(".layui-nav-third-child");
		if (thirdchild.is(":hidden")) {
			if ($(".mclick_span").text() == "全部收起") {
				$(this).parent().parent().next().find(".layui-nav-third-child").hide();
				$(this).parent().parent().next().find(".mselect_img").css('background-position', '-115px -245px');
				$(".mclick_span").text("全部展开");
				$("i.g-arrow-i").css({
					"border-color": "#56b282 transparent transparent",
					"top": "18px"
				});
				$(".mselect_img").removeClass('opened');
			} else {
				$(this).parent().parent().next().find(".layui-nav-third-child").show();
				$(this).parent().parent().next().find(".mselect_img").css('background-position', '-77px -245px');
				$(".mclick_span").text("全部收起");
				$("i.g-arrow-i").css({
					"border-color": "transparent transparent #56b282",
					"top": "13px"
				});
				$(".mselect_img").addClass('opened');
			};
		} else {
			$(this).parent().parent().next().find(".layui-nav-third-child").hide();
			$(this).parent().parent().next().find(".mselect_img").css('background-position', '-115px -245px');
			$(".mclick_span").text("全部展开");
			$("i.g-arrow-i").css({
				"border-color": "#56b282 transparent transparent",
				"top": "18px"
			});
			$(".mselect_img").removeClass('opened');
		};
	});
	// 鼠标移动改变a背景颜色
	$(".mselect_tit").mousemove(function () {
		$(this).css("background", "#f2f2f2");
	}).mouseout(function () {
		$(this).css("background", "transparent");
	});
	// 接收paper_1和view_original的值，隐藏目录
	window.addEventListener("message", function (event) {
		if (event.data = "text") {
			$(".mlayui-nav-child").hide();
			$(".bookmark1_layui-nav-child").hide();
		};
	}, false);
	// 传值给left
	var iframe = document.getElementById('childIframe');
	window.onload = function () {
		var m_navchild = $(".mlayui-nav-child").css("display");
		var o_navchild = $(".bookmark1_layui-nav-child").css("display");
		var obj = {
			child1: m_navchild,
			child2: o_navchild,
		};
		iframe.contentWindow.postMessage(obj, "*");
		// 关闭列表传值给left和查看原文
		$(".m_close").click(function () {
			$(".mlayui-nav-child").hide();
			$(".bookmark1_layui-nav-child").hide();
			var obj = {
				child1: m_navchild,
				child2: o_navchild,
			};
			iframe.contentWindow.postMessage(obj, "*");
		});
		window.addEventListener("message", function (event) {
			$(".mlayui-nav-child").css("display", event.data.state1);
			$(".bookmark1_layui-nav-child").css("display", event.data.state2);
		}, false);

	};
	// 动态获取宽高
	var w1 = screen.width;
	var w = document.documentElement.clientWidth;
	var h = document.documentElement.clientHeight;
	var shadeLeft = (w - 500) / 2;
	$("body").height(h);
	$(".layui-layer").css({
		"left": shadeLeft,
		"top": "60px"
	});
	$(".bookmark").css({
		"width": w * 0.29,
		"left": w * 0.7 - 30
	});
	$(".bookmark1").css({
		"width": w * 0.29,
		"right": 72
	});
	listHeight();
});
})(Report,jQuery);
function listHeight() {
	var h = document.documentElement.clientHeight;
	var show = $('.paper-header').css('display');
	if (show == 'block') {
		$(".overflow_ul").css("max-height", h - 240);
	};
	if (show == 'none') {
		$(".overflow_ul").css("max-height", h - 140);
	};
};
// 窗口拖拽
$(window).resize(function () {
	var w1 = screen.width;
	var w = document.documentElement.clientWidth;
	var h = document.documentElement.clientHeight;
	var shadeLeft = (w - 500) / 2;
	$("body").height(h);
	$(".layui-layer").css({
		"left": shadeLeft,
		"top": "60px"
	});
	$(".bookmark").css({
		"width": w * 0.29,
		"left": w * 0.7 - 30
	});
	$(".bookmark1").css({
		"max-width": w * 0.29,
		"right": 72
	});
	listHeight();
});