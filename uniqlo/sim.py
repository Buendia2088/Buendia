import time
import random
import json
from datetime import datetime

# 扩展数据库类
class FashionDatabase:
    def __init__(self):
        self.products = self._generate_products()
        self.inventory = self._generate_inventory()
        self.users = self._generate_users()
        self.staff = ["王店员", "李店员", "张店员"]
        self.rfid_tags = self._generate_rfid_tags()
        self.store_layout = self._generate_store_layout()
        
    def _generate_products(self):
        products = {
            "U1234": {"name": "HEATTECH圆领T恤", "category": "上衣", "colors": ["白色", "黑色", "灰色"], 
                     "price": 79, "tech": "保暖科技", "material": "聚酯纤维"},
            "U5678": {"name": "弹力牛仔裤", "category": "下装", "colors": ["深蓝", "浅蓝", "黑色"], 
                     "price": 249, "series": "基础款", "material": "棉"},
            "U9012": {"name": "轻型羽绒夹克", "category": "外套", "colors": ["黑色", "海军蓝", "米色"], 
                     "price": 599, "series": "保暖系列", "material": "尼龙"},
            "U3456": {"name": "法兰绒衬衫", "category": "上衣", "colors": ["格纹红", "格纹蓝", "格纹绿"], 
                     "price": 149, "series": "经典系列", "material": "棉"},
            "U7890": {"name": "运动束脚裤", "category": "下装", "colors": ["黑色", "灰色", "藏青"], 
                     "price": 199, "series": "运动系列", "material": "聚酯纤维"},
            "U1122": {"name": "AIRism圆领T恤", "category": "上衣", "colors": ["白色", "黑色", "藏青"], 
                     "price": 79, "tech": "凉感科技", "eco": True, "material": "再生聚酯纤维"},
            "U3344": {"name": "摇粒绒外套", "category": "外套", "colors": ["米色", "深灰", "藏青"], 
                     "price": 299, "series": "环保系列", "eco": True, "material": "再生聚酯纤维"},
            "U5566": {"name": "高腰直筒裙", "category": "下装", "colors": ["黑色", "卡其", "酒红"], 
                     "price": 199, "series": "Peace for All", "special": True},
            "U7788": {"name": "UNIQLO UT联名T恤", "category": "上衣", "colors": ["白色", "黑色", "限量色"], 
                     "price": 99, "series": "设计师联名", "special": True},
        }
        
        # 添加搭配关系
        products["U1234"]["matches"] = ["U5678", "U9012", "U5566"]
        products["U5678"]["matches"] = ["U1234", "U1122", "U9012", "U3344"]
        products["U9012"]["matches"] = ["U1234", "U5678", "U3456"]
        products["U3456"]["matches"] = ["U7890", "U5678", "U5566"]
        products["U7890"]["matches"] = ["U3456", "U1122", "U3344"]
        products["U1122"]["matches"] = ["U7890", "U5678", "U5566"]
        products["U3344"]["matches"] = ["U1122", "U7890", "U5566"]
        products["U5566"]["matches"] = ["U1122", "U1234", "U3344"]
        products["U7788"]["matches"] = ["U5678", "U7890"]
        
        # 添加环保数据
        self.eco_impact = {
            "再生聚酯纤维": 0.25  # kg CO2 减排/件
        }
        
        return products
    
    def _generate_inventory(self):
        inventory = {}
        for product_id in self.products:
            for color in self.products[product_id]["colors"]:
                for size in ["S", "M", "L", "XL"]:
                    # 随机库存数量 0-5
                    inventory[f"{product_id}-{color}-{size}"] = random.randint(0, 5)
        return inventory
    
    def _generate_users(self):
        return {
            "user-001": {"name": "张明", "preferences": ["简约", "舒适"], "size": "M", 
                        "past_purchases": ["U1234", "U5678"], "eco_conscious": True},
            "user-002": {"name": "李娜", "preferences": ["时尚", "潮流"], "size": "S", 
                        "past_purchases": ["U9012", "U5566"], "eco_conscious": False},
            "user-003": {"name": "王伟", "preferences": ["商务", "休闲"], "size": "L", 
                        "past_purchases": ["U3456", "U7890"], "eco_conscious": True},
        }
    
    def _generate_rfid_tags(self):
        tags = []
        for product_id, product in self.products.items():
            for color in product["colors"]:
                for size in ["S", "M", "L", "XL"]:
                    tags.append(f"{product_id}-{color}-{size}")
        return tags
    
    def _generate_store_layout(self):
        """生成店铺布局"""
        return {
            "A": {"name": "男装区", "items": ["U1234", "U5678", "U9012", "U3456"]},
            "B": {"name": "女装区", "items": ["U5566", "U7890", "U1122", "U3344"]},
            "C": {"name": "联名专区", "items": ["U7788"]},
            "D": {"name": "试衣间走廊", "items": []},
            "E": {"name": "环保系列区", "items": ["U1122", "U3344"]},
            "F": {"name": "特惠区", "items": []}
        }
    
    def get_product_location(self, product_id):
        """获取商品所在区域"""
        for area, area_data in self.store_layout.items():
            if product_id in area_data["items"]:
                return area, area_data["name"]
        return "F", "特惠区"  # 默认
    
    def get_route_to_location(self, from_area, to_area):
        """生成路线指引"""
        routes = {
            ("A", "B"): "左转穿过中央走廊",
            ("A", "C"): "直走至尽头右转",
            ("A", "E"): "穿过男装区，右转至环保区",
            ("B", "A"): "右转穿过中央走廊",
            ("B", "C"): "直走至尽头左转",
            ("B", "E"): "穿过女装区，左转至环保区",
            ("C", "A"): "返回中央走廊后左转",
            ("C", "B"): "返回中央走廊后右转",
            ("E", "A"): "穿过环保区，左转至男装区",
            ("E", "B"): "穿过环保区，右转至女装区",
        }
        return routes.get((from_area, to_area), "请跟随地面指示前往")
    
    def get_eco_impact(self, material):
        """获取环保影响数据"""
        return self.eco_impact.get(material, 0)
    
    def get_random_rfid(self):
        return random.choice(self.rfid_tags)
    
    def get_product_info(self, product_id):
        return self.products.get(product_id)
    
    def get_inventory(self, product_id, color=None, size=None):
        results = {}
        for key, quantity in self.inventory.items():
            if key.startswith(product_id):
                if (color is None or color in key) and (size is None or size in key):
                    _, color, size = key.split('-')
                    results[f"{color}-{size}"] = quantity
        return results
    
    def get_user_info(self, user_id):
        return self.users.get(user_id)
    
    def get_random_user(self):
        return random.choice(list(self.users.keys()))
    
    def get_available_staff(self):
        return random.choice(self.staff)
    
    def update_inventory(self, product_id, color, size, change=-1):
        key = f"{product_id}-{color}-{size}"
        if key in self.inventory and self.inventory[key] > 0:
            self.inventory[key] += change
            return True
        return False

# 增强版AI搭配引擎
class StylingEngine:
    def __init__(self, db):
        self.db = db
        self.special_series_info = {
            "Peace for All": {
                "title": "Peace for All 系列",
                "description": "本系列旨在传递和平与包容的理念，部分收益将捐赠给联合国儿童基金会。",
                "image": "🕊️"
            },
            "设计师联名": {
                "title": "UNIQLO UT 设计师联名",
                "description": "与世界知名艺术家和设计师合作，打造独特时尚单品。",
                "image": "🎨"
            },
            "环保系列": {
                "title": "环保再生系列",
                "description": "使用再生材料制成，减少环境负担。",
                "image": "♻️"
            }
        }
    
    def get_recommendations(self, product_id, user_id=None):
        product = self.db.get_product_info(product_id)
        if not product:
            return []
        
        # 基础搭配推荐
        recommendations = []
        for match_id in product.get("matches", []):
            match_product = self.db.get_product_info(match_id)
            if match_product:
                # 获取商品位置
                area, area_name = self.db.get_product_location(match_id)
                
                recommendations.append({
                    "product_id": match_id,
                    "name": match_product["name"],
                    "category": match_product["category"],
                    "price": match_product["price"],
                    "reason": f"与{product['name']}搭配",
                    "location": area_name,
                    "area": area
                })
        
        # 个性化推荐 (基于用户偏好)
        if user_id:
            user = self.db.get_user_info(user_id)
            if user and user["preferences"]:
                # 根据用户偏好筛选
                for pid, p in self.db.products.items():
                    if pid != product_id and pid not in [r["product_id"] for r in recommendations]:
                        reason = ""
                        if "简约" in user["preferences"] and ("法兰绒" not in p["name"] and "格纹" not in p["name"]):
                            reason = "符合您简约的偏好"
                        elif "时尚" in user["preferences"] and ("羽绒" in p["name"] or "高腰" in p["name"] or "联名" in p["name"]):
                            reason = "符合您时尚的偏好"
                        elif "商务" in user["preferences"] and ("衬衫" in p["name"] or "直筒" in p["name"]):
                            reason = "符合您商务的偏好"
                        elif "舒适" in user["preferences"] and ("摇粒绒" in p["name"] or "AIRism" in p["name"] or "HEATTECH" in p["name"]):
                            reason = "符合您舒适的偏好"
                            
                        if reason:
                            # 获取商品位置
                            area, area_name = self.db.get_product_location(pid)
                            
                            recommendations.append({
                                "product_id": pid,
                                "name": p["name"],
                                "category": p["category"],
                                "price": p["price"],
                                "reason": reason,
                                "location": area_name,
                                "area": area
                            })
        
        # 去重
        seen = set()
        unique_recommendations = []
        for r in recommendations:
            if r["product_id"] not in seen:
                seen.add(r["product_id"])
                unique_recommendations.append(r)
        
        return unique_recommendations[:4]  # 返回最多4个推荐
    
    def get_special_series_info(self, series_name):
        """获取特殊系列信息"""
        return self.special_series_info.get(series_name, {
            "title": series_name,
            "description": "精选系列",
            "image": "✨"
        })

# 增强版智能试衣间系统
class SmartFittingRoom:
    def __init__(self, room_id, db):
        self.room_id = room_id
        self.db = db
        self.styling_engine = StylingEngine(db)
        self.current_user = None
        self.current_items = []
        self.staff_notifications = []
        self.current_area = "D"  # 试衣间在D区
        self.total_eco_impact = 0.0  # 累计环保影响
    
    def detect_rfid(self, rfid_tag):
        """解析RFID标签获取商品信息"""
        parts = rfid_tag.split('-')
        if len(parts) < 3:
            return None
        
        product_id = parts[0]
        color = parts[1]
        size = parts[2]
        
        product_info = self.db.get_product_info(product_id)
        if not product_info:
            return None
        
        # 如果是环保材料，计算环保影响
        eco_impact = 0.0
        if product_info.get("eco"):
            material = product_info.get("material", "")
            eco_impact = self.db.get_eco_impact(material)
            self.total_eco_impact += eco_impact
        
        return {
            "product_id": product_id,
            "name": product_info["name"],
            "color": color,
            "size": size,
            "price": product_info["price"],
            "eco_impact": eco_impact,
            "series": product_info.get("series", ""),
            "special": product_info.get("special", False)
        }
    
    def user_enter(self, user_id=None):
        """用户进入试衣间"""
        if not user_id:
            user_id = self.db.get_random_user()
        self.current_user = self.db.get_user_info(user_id)
        self.current_items = []
        self.total_eco_impact = 0.0
        print(f"\n[试衣间 {self.room_id}] 用户 {self.current_user['name']} 进入")
        print(f"偏好: {', '.join(self.current_user['preferences'])} | 常用尺码: {self.current_user['size']}")
        
        # 显示欢迎信息
        if self.current_user.get("eco_conscious", False):
            print("\n🌱 感谢您选择环保产品！您的每次环保选择都在为地球做出贡献")
        
        return True
    
    def user_add_item(self, rfid_tag):
        """用户添加衣物到试衣间"""
        if not self.current_user:
            print(">> 请先进入试衣间")
            return False
            
        item = self.detect_rfid(rfid_tag)
        if not item:
            print(">> 无法识别商品")
            return False
        
        self.current_items.append(item)
        print(f">> 添加商品: {item['name']} ({item['color']}, {item['size']}) ¥{item['price']}")
        
        # 显示特殊系列信息
        if item.get("special", False) and item.get("series", ""):
            series_info = self.styling_engine.get_special_series_info(item["series"])
            print(f"\n✨ {series_info['image']} {series_info['title']} ✨")
            print(series_info["description"])
            print("="*30)
        
        # 显示环保贡献
        if item["eco_impact"] > 0:
            print(f"\n♻️ 环保贡献: 您选择的 {item['name']} 减少了 {item['eco_impact']:.2f} kg CO₂ 排放")
            print(f"累计环保贡献: {self.total_eco_impact:.2f} kg CO₂ 排放")
            print("="*30)
        
        return True
    
    def show_inventory(self, product_id, color):
        """显示其他尺寸库存"""
        if not product_id or not color:
            if not self.current_items:
                print(">> 试衣间没有商品")
                return
            item = self.current_items[-1]  # 默认显示最后添加的商品
            product_id = item["product_id"]
            color = item["color"]
        
        inventory = self.db.get_inventory(product_id, color)
        product_name = self.db.get_product_info(product_id)["name"]
        
        print(f"\n=== {product_name} ({color}) 尺寸库存 ===")
        for size_qty in inventory.items():
            print(f"尺寸 {size_qty[0]}: {'✅有货' if size_qty[1] > 0 else '❌缺货'}")
        print("="*30)
        return True
    
    def show_recommendations(self, product_id=None):
        """显示搭配推荐"""
        if not self.current_user:
            print(">> 请先进入试衣间")
            return False
            
        if not product_id:
            if not self.current_items:
                print(">> 试衣间没有商品")
                return False
            item = self.current_items[-1]  # 默认显示最后添加的商品
            product_id = item["product_id"]
        
        user_id = [k for k, v in self.db.users.items() if v == self.current_user][0]
        recommendations = self.styling_engine.get_recommendations(product_id, user_id)
        
        if not recommendations:
            print(">> 暂无推荐")
            return False
        
        product_name = self.db.get_product_info(product_id)["name"]
        print(f"\n✨ 智能搭配推荐 [{product_name}]:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['name']} ({rec['category']}) ¥{rec['price']}")
            print(f"   → {rec['reason']}")
            print(f"   位置: {rec['location']} (路线: {self.db.get_route_to_location(self.current_area, rec['area'])})")
        print("="*30)
        return True
    
    def request_item(self, product_id=None, color=None, size=None):
        """请求其他尺寸或商品"""
        if not self.current_user:
            print(">> 请先进入试衣间")
            return False
        
        # 如果没有提供参数，使用最后添加的商品
        if not product_id or not color or not size:
            if not self.current_items:
                print(">> 试衣间没有商品")
                return False
                
            item = self.current_items[-1]
            product_id = item["product_id"]
            color = item["color"]
            
            # 显示库存并让用户选择尺寸
            self.show_inventory(product_id, color)
            size = input(">> 请输入您想尝试的尺寸 (如 M): ").strip().upper()
            if size not in ["S", "M", "L", "XL"]:
                print(">> 无效尺寸")
                return False
        
        # 检查库存
        key = f"{product_id}-{color}-{size}"
        inventory = self.db.get_inventory(product_id, color, size)
        
        if not inventory or list(inventory.values())[0] <= 0:
            print(f">> {color}-{size} 缺货")
            return False
        
        # 获取商品位置
        area, area_name = self.db.get_product_location(product_id)
        route_info = self.db.get_route_to_location(self.current_area, area)
        
        # 判断是否提供送货服务（店员空闲时）
        staff_busy = len([r for r in self.staff_notifications if r["status"] == "待处理"]) > 2
        delivery_offered = not staff_busy
        
        print(f"\n📌 该商品位于: {area_name}")
        print(f"🚶 路线指引: {route_info}")
        
        if delivery_offered:
            print("\n👔 店员目前有空，可以为您送货到试衣间！")
            choice = input(">> 是否需要店员送货？(y/n): ").strip().lower()
            if choice == 'y':
                return self._create_delivery_request(product_id, color, size, area_name)
        
        return self._create_self_pickup_request(product_id, color, size, area_name, route_info)
    
    def _create_delivery_request(self, product_id, color, size, location):
        """创建送货请求"""
        staff = self.db.get_available_staff()
        request_time = datetime.now().strftime("%H:%M:%S")
        request_id = f"DEL-{time.time_ns() % 1000000}"
        
        notification = {
            "id": request_id,
            "room": self.room_id,
            "user": self.current_user["name"],
            "product": self.db.get_product_info(product_id)["name"],
            "color": color,
            "size": size,
            "time": request_time,
            "staff": staff,
            "status": "待处理",
            "type": "送货",
            "location": location
        }
        
        self.staff_notifications.append(notification)
        print(f"\n✅ 已通知店员 ({staff}) 为您送货: {color}色 {size}码")
        print(f"预计送达时间: 3-5分钟")
        return True
    
    def _create_self_pickup_request(self, product_id, color, size, location, route):
        """创建自取请求"""
        staff = self.db.get_available_staff()
        request_time = datetime.now().strftime("%H:%M:%S")
        request_id = f"REQ-{time.time_ns() % 1000000}"
        
        notification = {
            "id": request_id,
            "room": self.room_id,
            "user": self.current_user["name"],
            "product": self.db.get_product_info(product_id)["name"],
            "color": color,
            "size": size,
            "time": request_time,
            "staff": staff,
            "status": "待处理",
            "type": "自取",
            "location": location,
            "route": route
        }
        
        self.staff_notifications.append(notification)
        print(f"\n✅ 已保存您的请求: {color}色 {size}码")
        print(f"您可以随时前往 {location} 取货")
        print(f"路线指引: {route}")
        return True
    
    def process_requests(self):
        """处理所有待处理请求（模拟店员操作）"""
        if not self.staff_notifications:
            print(">> 没有待处理请求")
            return False
        
        completed = 0
        for req in self.staff_notifications[:]:
            if req["status"] == "待处理":
                req["status"] = "已完成"
                req["complete_time"] = datetime.now().strftime("%H:%M:%S")
                
                # 更新库存
                product_id = next((pid for pid, p in self.db.products.items() 
                                  if p["name"] == req["product"]), None)
                if product_id:
                    self.db.update_inventory(product_id, req["color"], req["size"])
                
                if req["type"] == "送货":
                    print(f">> 店员 {req['staff']} 已将 {req['product']} ({req['color']}-{req['size']}) 送达试衣间")
                else:
                    print(f">> 店员 {req['staff']} 已准备 {req['product']} ({req['color']}-{req['size']}) 在 {req['location']}")
                
                completed += 1
        
        return completed > 0
    
    def show_current_items(self):
        """显示当前试衣间内的商品"""
        if not self.current_user:
            print(">> 请先进入试衣间")
            return False
            
        if not self.current_items:
            print(">> 试衣间没有商品")
            return False
        
        print(f"\n🛍️ 当前试衣间商品 ({len(self.current_items)}件):")
        total_eco = 0.0
        for i, item in enumerate(self.current_items, 1):
            eco_info = f" | ♻️ -{item['eco_impact']:.2f}kg CO₂" if item['eco_impact'] > 0 else ""
            print(f"{i}. {item['name']} ({item['color']}, {item['size']}) ¥{item['price']}{eco_info}")
            total_eco += item['eco_impact']
        
        if total_eco > 0:
            print(f"\n🌍 累计环保贡献: 减少 {total_eco:.2f} kg CO₂ 排放")
            print("相当于节省了约 {:.1f} 升汽油的碳排放".format(total_eco * 2.3))
        
        print("="*30)
        return True
    
    def show_pending_requests(self):
        """显示待处理请求"""
        if not self.staff_notifications:
            print(">> 没有待处理请求")
            return False
        
        pending = [r for r in self.staff_notifications if r["status"] == "待处理"]
        if not pending:
            print(">> 没有待处理请求")
            return False
        
        print("\n⏱️ 待处理请求:")
        for i, req in enumerate(pending, 1):
            print(f"{i}. {req['product']} ({req['color']}-{req['size']})")
            print(f"   类型: {req['type']} | 位置: {req.get('location', '未知')}")
            print(f"   请求时间: {req['time']} | 店员: {req['staff']}")
            if req["type"] == "自取":
                print(f"   路线指引: {req.get('route', '请咨询店员')}")
        print("="*30)
        return True
    
    def user_exit(self):
        """用户离开试衣间"""
        if not self.current_user:
            print(">> 没有用户在试衣间")
            return False
            
        user_name = self.current_user["name"]
        
        # 显示购物总结
        if self.current_items:
            print("\n感谢您试穿！以下商品已加入您的试穿记录:")
            for item in self.current_items:
                print(f"- {item['name']} ({item['color']}, {item['size']})")
            
            # 环保贡献总结
            if self.total_eco_impact > 0:
                print(f"\n🌱 您的环保选择减少了 {self.total_eco_impact:.2f} kg CO₂ 排放")
                print("相当于种植了 {:.1f} 棵树一年的碳吸收量".format(self.total_eco_impact * 0.5))
        
        self.current_user = None
        self.current_items = []
        print(f"\n[试衣间 {self.room_id}] 用户 {user_name} 离开")
        return True

# 增强交互式系统
class FittingRoomSimulator:
    def __init__(self):
        self.db = FashionDatabase()
        self.fitting_room = SmartFittingRoom("A-01", self.db)
        self.running = True
    
    def display_menu(self):
        """显示主菜单"""
        print("\n" + "="*50)
        print("优衣库智能试衣间系统")
        print("="*50)
        
        if self.fitting_room.current_user:
            user = self.fitting_room.current_user
            print(f"当前用户: {user['name']} | 偏好: {', '.join(user['preferences'])}")
            print(f"试衣间商品: {len(self.fitting_room.current_items)}件")
            
            # 显示环保贡献
            if self.fitting_room.total_eco_impact > 0:

                print(f"环保贡献: 减少 {self.fitting_room.total_eco_impact:.2f} kg CO₂")
        else:
            print("当前状态: 试衣间空闲")
        
        print("\n主菜单:")
        print("1. 用户进入试衣间")
        print("2. 添加衣物到试衣间")
        print("3. 查看当前衣物")
        print("4. 查看库存状态")
        print("5. 查看搭配推荐")
        print("6. 请求其他尺寸")
        print("7. 查看待处理请求")
        print("8. 处理请求(店员)")
        print("9. 用户离开试衣间")
        print("0. 显示所有商品")
        print("v. 查看店铺地图")
        print("q. 退出系统")
    
    def show_all_products(self):
        """显示所有商品信息"""
        print("\n所有商品列表:")
        for i, (pid, product) in enumerate(self.db.products.items(), 1):
            # 显示环保标志
            eco_flag = " ♻️" if product.get("eco") else ""
            special_flag = " ✨" if product.get("special") else ""
            
            print(f"{i}. {product['name']}{eco_flag}{special_flag} ({product['category']}) ¥{product['price']}")
            print(f"   颜色: {', '.join(product['colors'])}")
            
            # 显示位置信息
            area, area_name = self.db.get_product_location(pid)
            print(f"   位置: {area_name} ({area})")
        print("="*50)
    
    def show_store_map(self):
        """显示店铺地图"""
        print("\n店铺地图:")
        print("A: 男装区     B: 女装区")
        print("C: 联名专区   D: 试衣间走廊")
        print("E: 环保系列区 F: 特惠区")
        print("\n当前位置: 试衣间 (D区)")
        print("="*50)
    
    def select_product(self):
        """让用户选择商品"""
        products = list(self.db.products.items())
        self.show_all_products()
        
        try:
            choice = int(input(">> 请选择商品编号: "))
            if 1 <= choice <= len(products):
                product_id, product = products[choice-1]
                
                # 显示颜色选项
                print(f"\n可选颜色: {', '.join(product['colors'])}")
                color = input(">> 请选择颜色: ").strip()
                if color not in product["colors"]:
                    print(">> 无效颜色选择")
                    return None
                
                # 显示尺寸选项
                print("可选尺寸: S, M, L, XL")
                size = input(">> 请选择尺寸: ").strip().upper()
                if size not in ["S", "M", "L", "XL"]:
                    print(">> 无效尺寸")
                    return None
                
                return f"{product_id}-{color}-{size}"
            else:
                print(">> 无效选择")
        except ValueError:
            print(">> 请输入数字")
        
        return None
    
    def run(self):
        """运行交互式系统"""
        print("="*50)
        print("优衣库智能试衣间系统模拟")
        print("="*50)
        print("提示: 输入 'q' 退出系统\n")
        
        while self.running:
            self.display_menu()
            choice = input("\n>> 请选择操作: ").strip().lower()
            
            if choice == 'q':
                self.running = False
                print("系统已退出")
                break
            
            try:
                if choice == '1':  # 用户进入
                    self.fitting_room.user_enter()
                
                elif choice == '2':  # 添加衣物
                    rfid = self.select_product()
                    if rfid:
                        self.fitting_room.user_add_item(rfid)
                
                elif choice == '3':  # 查看当前衣物
                    self.fitting_room.show_current_items()
                
                elif choice == '4':  # 查看库存
                    if self.fitting_room.current_items:
                        self.fitting_room.show_inventory(None, None)
                    else:
                        print(">> 试衣间没有商品")
                
                elif choice == '5':  # 查看推荐
                    self.fitting_room.show_recommendations()
                
                elif choice == '6':  # 请求其他尺寸
                    self.fitting_room.request_item()
                
                elif choice == '7':  # 查看待处理请求
                    self.fitting_room.show_pending_requests()
                
                elif choice == '8':  # 处理请求
                    self.fitting_room.process_requests()
                
                elif choice == '9':  # 用户离开
                    self.fitting_room.user_exit()
                
                elif choice == '0':  # 显示所有商品
                    self.show_all_products()
                
                elif choice == 'v':  # 查看店铺地图
                    self.show_store_map()
                
                else:
                    print(">> 无效选择，请重新输入")
            
            except Exception as e:
                print(f">> 发生错误: {str(e)}")
            
            # 每次操作后暂停一下
            input("\n按 Enter 键继续...")

# 启动系统
if __name__ == "__main__":
    simulator = FittingRoomSimulator()
    simulator.run()