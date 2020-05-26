/**
 * Created by haohonglong on 17/8/8.
 */

if(!GRN_P){var GRN_P='Report';}
(function(global,namespace,factory,undefined){
    'use strict';
    global[namespace] = factory(global,namespace);
})(typeof window !== "undefined" ? window : this,GRN_P,function(W,namespace,undefined){
    'use strict';
    return {};
});
/**
 * 缓存
 */
(function(System,$){
    var __this__ = null;
    function isset(s){
        return (typeof s !== "undefined" &&  s !== null);
    }
    var cache = [],cache_name='cache',
        myStorage = null;

    function isStorage(){
        if(typeof(myStorage) !== "undefined") {
            return true;
        } else {
            return false;
        }
    }
    function set(value,name){
        value = value || cache;
        name = name || cache_name;
        if(isStorage()){
            remove();
            myStorage.setItem(name,JSON.stringify(value));
        }
    }
    function remove(name){
        name = name || cache_name;
        if(isStorage()){
            myStorage.removeItem(name);
        }
    }

    function clear() {
        if(isStorage()){
            myStorage.clear();
        }else{
            cache = [];
        }

    }

    function get(name) {
        name = name || cache_name;
        if(isStorage()){
            return (JSON.parse(myStorage.getItem(name))) || cache;
        }else{
            return cache;
        }

    }

    /**
     *
     * @param name
     * @param type 存储类型
     * @constructor
     */
    function Cache(name,type){
        __this__=this;
        this.caches = [];
        cache_name = name || 'cache';
        this.cache_name = cache_name;
        this.myStorage = type || localStorage;
    }
    Cache.prototype = {
        'constructor':Cache,
        '_className':'Cache',
        'init':function(){
            myStorage  = this.myStorage;
            cache_name = this.cache_name;
            cache = get();
        },
        'cache':function(key,value,callback){
            this.init();
            cache = get();
            var index = this.exist(key,value);
            if($.isFunction(callback)){
                callback.call(this,index);
            }
            return index;
        },
        'set':function(Obj,key,value){
            this.init();
            Obj[key] = value;
            cache.push(Obj);
            set();

        },
        'update':function(index,Obj){
            this.init();
            cache[index] = Obj;
            set();
        },
        'get':function(index){
            this.init();
            return isset(index) ? cache[index] : cache;
        },
        'exist':function(key,value){
            this.init();
            for(var i=0,len=cache.length;i<len;i++){
                if((key in cache[i]) && (value == cache[i][key])){
                    return i;
                }
            }
            return -1;
        },

        'clear':function(){
            this.init();
            this.remove();
            clear();
        },
        'remove':function(index){
            this.init();
            if(index){
                if (index > -1 && index < cache.length-1) {
                    cache.splice(index, 1);
                    // delete cache[index];
                }
            }else{
                cache = [];
            }
            set();
        }
    };
    System.Cache = Cache;
})(Report,jQuery);


/**
 * 页面操作状态保存进缓存中
 */
(function(System,$){
    'use strict';
    var storage = window.localStorage;
    function getObj(id) {
        id=id+"_s";
        var obj = JSON.parse(storage.getItem(id))
        return obj
    }
    function setObj(id, obj) {
        id=id+"_s";
        storage.setItem(id,JSON.stringify(obj))
    }
    function operateState(){}
    /**
     * 以id为key保存操作状态对象
     * @param id 文档的id
     */
    operateState.saveId = function(id){
        var defaults = {
            "red": "",
            "LineState": "",
            "HeaderState": '',
            "aLink": "htmls/detail_report.html",
            "tabName": "" 
        };
        id=id+"_s";
        if (storage.getItem(id) === null) {
            storage.setItem(id,JSON.stringify(defaults));
        } else {
            return
        }
        
    };
    /**
     * 存取头部状态
     * @param id id
     * @param show 头部状态
     */
    operateState.saveHeaderState = function(id, show) {
        var obj = getObj(id);
        obj.HeaderState = show;
        setObj(id, obj);
    };
    operateState.getHeaderState = function(id) {
        var obj = getObj(id);
        return obj.HeaderState;
    };
    /**
     * 存取a的href
     * @param id id
     * @param a_Attr 链接
     */
    operateState.saveALink = function(id, a_Attr) {
        var obj = getObj(id);
        obj.aLink = a_Attr;
        setObj(id, obj);
    };
    operateState.getALink = function(id) {
        var obj = getObj(id);
        return obj.aLink;
    };
    /**
     * 存取tab
     * @param id id
     * @param tab_name tab
     */
    operateState.saveTabName = function(id, tab_name) {
        var obj = getObj(id);
        obj.tabName = tab_name;
        setObj(id, obj);
    };
    operateState.getTabName = function(id) {
        var obj = getObj(id);
        return obj.tabName;
    };

    System.operateState = operateState
})(Report,jQuery);



(function(System,$){
    'use strict';
    var WIN_H,MIN_HEIGHT = "600";
    var cache = null;
    function getCache(){
        if(!cache){cache = new System.Cache(System.report_id,localStorage);}
    }
    function Paper(){}
    /**
     *设置main 页面的 iframe 高度
     * @param D
     * @returns {Function}
     */
    Paper.setMainIframeHeight=function(D){
        var defaults ={
            "header":'#m-header',
            "nav":'#m-nav',
            "content":'#m-content'
        };
        D = $.isPlainObject(D) ? $.extend(defaults,D) : defaults;
        var $iframe,$header,header_h,$nav,nav_h;
        function initIframe() {
            var height = WIN_H - (header_h + nav_h);
            // if (height < MIN_HEIGHT) {
            //     height = MIN_HEIGHT;
            // }

            $iframe.height(height);
            $iframe.find('iframe').height(height);
        }
        return function(){
            $header = $(D.header);
            header_h = $header.height();
            if(!header_h){header_h = 0;}
            $nav = $(D.nav);
            nav_h = $nav.height();
            if(!nav_h){nav_h = 0;}
            $iframe = $(D.content);
            WIN_H = $(window).height();
            initIframe();

        };

    };
    /**
     * 段落修改
     * @param D
     * notice: this 触发的按钮对象元素
     *
     */
    Paper.sectionEdit=function(D){
        getCache();
        var defaults ={
            //"btn":"[tpl-section=btn]",
            "textWarp":"[tpl-section=warp]",
            "text":"[tpl-section=text]",
            "badge":"[tpl-section=badge]",
            "box":"[tpl-section=box]",
            "textarea":"[tpl-section=textarea]",
            "template":'script[type="text/html"][template="section"]'
        };
        D = $.isPlainObject(D) ? $.extend(defaults,D) : defaults;
        var old_dom=null;
        return function(){
            //if(old_dom === this){return;}
            old_dom = this;
            var text="";
            var $this = $(this);
            var $warp = $this.closest(D.textWarp);
            var id = $warp.find(D.badge).data('id');
            var $text = $warp.find(D.text);
            var $box = $warp.find(D.box);
            $box.html($(D.template).html());
            var $textarea = $warp.find(D.textarea);
            cache.cache('id',id,function (index) {
                if(-1 === index){
                    text = $text.text();
                }else{
                    var Obj = cache.get(index);
                    text = Obj.text;

                }
                $textarea.val($.trim(text));
                $textarea.text($textarea.val()).focus();
            });
        };

    };

    /**
     * 段落保存
     * @param D.id
     * @param D.text
     * notice: this 触发的按钮对象元素
     *
     */
    Paper.sectionSave=function(D){
        getCache();
        var defaults ={
            'warp':'[tpl-section="warp"]',
            'box':'[tpl-section="box"]',
            'badge':'[tpl-section="badge"]',
            'textarea':'[tpl-section="textarea"]'
        };
        D = $.isPlainObject(D) ? $.extend(defaults,D) : defaults;
        var old_dom=null;
        return function(){
            //if(old_dom === this){return;}
            var $warp = $(this).closest(D.warp);
            var text = $warp.find(D.textarea).val();
            text = $.trim(text);
            var id = $warp.find(D.badge).data('id');
            cache.cache('id', id,function (index) {
                if(-1 === index){
                    cache.set({'text': text},'id',id);
                }else{
                    cache.update(index,{'id': id,'text': text});
                }
            });
            $warp.find(D.box).html('');
        };

    };


    /**
     * 选项卡
     * @param D
     * @returns {Function}
     */
    Paper.tab=function(){
        var old_dom=null;
        return function(D){
            if(old_dom === this){return;}
            var defaults ={
                "li":"li",
                "section":'[tab="section"]',
                "active":'active',
                "callback":function(){}
            };
            D = $.isPlainObject(D) ? $.extend(defaults,D) : defaults;
            old_dom = this;
            var $this = $(this);
            $this.parent().find(D.li).removeClass(D.active);
            $this.addClass(D.active);
            var id = $this.data('id');
            if(!id){return;}
            var ids = id.toString().split(',');
            var $section = $(D.section);
            $section.hide();
            $section.each(function(){
                var $this = $(this);
                var id = $this.data('id');
                if($.inArray(id.toString(),ids) !== -1){
                    D.callback.call(this);
                    $this.show();
                }
            });

        };
    };


    /**
     * [eve-toggle=btn]
     */
    Paper.toggle=function(show_fn,hide_fn){
        var $section = $(this).closest('[eve-toggle=warp]').find('[eve-toggle=section]');
        if($section.css("display") === "none"){
            if($.isFunction(show_fn)){show_fn();}
            $section.show();
        }else{
            if($.isFunction(hide_fn)){hide_fn();}
            $section.hide();
        }
    };

    System.Paper = Paper;
})(Report,jQuery);

(function(System,$){
    'use strict';
    
})(Report,jQuery);

// 使用or取消辅助线
var flag = true;
var isAddLine = 'N'
function  auxiliaryLine(){
    if(flag){
        $('.addLine').html('取消辅助线');
        $('.red').css('border-bottom','1px solid #f12828');
        $('.orange').css('border-bottom','1px dashed #f39800');
        isAddLine = 'Y'
        return flag = false;
    }else{
        $('.addLine').html('使用辅助线');
        $('.red').css('border-bottom','0px');
        $('.orange').css('border-bottom','0px');
        isAddLine = 'N'
        return flag = true;
    }
}
$('.addLine').click(function(){
    auxiliaryLine();
})

// 分页传值
$('.pagination > li > a').click(function(e){
    e.preventDefault();
    location.href = $(this).attr('href') + '?isAddLine=' + isAddLine;
});

// 取值执行判断
var theRequest = GetRequest()
if(theRequest.isAddLine === 'Y'){
    auxiliaryLine()
}
function GetRequest() {
    var url = location.search;
    var theRequest = new Object();
    if (url.indexOf("?") != -1) {
        var str = url.substr(1);
        strs = str.split("&");
        for(var i = 0; i < strs.length; i ++) {
            theRequest[strs[i].split("=")[0]] = unescape(strs[i].split("=")[1]);
        }
    }
    return theRequest;
}