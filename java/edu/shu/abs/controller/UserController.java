package edu.shu.abs.controller;

import edu.shu.abs.common.Result;
import edu.shu.abs.common.StringUtils;
import edu.shu.abs.common.authentication.UserInfo;
import edu.shu.abs.constant.Role;
import edu.shu.abs.entity.User;
import edu.shu.abs.entity.Work;
import edu.shu.abs.service.HistoryUserWorkService;
import edu.shu.abs.service.UserService;
import edu.shu.abs.service.WorkService;
import edu.shu.abs.vo.user.UserInfoVo;
import edu.shu.abs.vo.user.UserLoginVo;
import edu.shu.abs.vo.user.UserPrivacyVo;
import edu.shu.abs.vo.user.UserRegisterVo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpSession;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
@Controller
@RequestMapping("/user")
public class UserController {
    @Autowired
    private UserService userService;
    
    @Autowired
    private HistoryUserWorkService historyUserWorkService;
    
    @Autowired
    private WorkService workService;
 
    @RequestMapping("/index")
    public String index() {
        return "index";
    }
 
    
    
    @ResponseBody
    @PostMapping("/login")
    public Result login(@RequestParam String username,
                          @RequestParam String password,
                          HttpSession session) {
        UserLoginVo user = new UserLoginVo();
        user.setUsername(username);
        user.setPassword(password);
        String token = userService.login(user);
        session.setAttribute("token", token);
        //角色判断
        if (username.equals("admin")&& password.equals("admin")) {
            return Result.success().data("role", Role.ADMIN.getKey());
        }
        return Result.success().data("role", Role.NORMAL_USER.getKey());
    }
    
    @RequestMapping("/home")
    public String home() {
        return "home";
    }
    
    @RequestMapping("/index/homepage")
    public String homepage() {
        return "homepage";
    }
    
    @RequestMapping("/favorites")
    public String favorites() {
        return "favorites";
    }
    
    @RequestMapping("/search")
    public String search() {
        return "search";
    }
    
    
    @RequestMapping("/toreg")
    public String toreg() {
        return "register";
    }
    
    @RequestMapping("/tosearch")
    public String tosearch() {
        return "tosearch";
    }
    
    @RequestMapping("/person")
    public String person() {
        return "person";
    }
    
    @RequestMapping("/admin")
    public String admin() {
        return "admin";
    }
    
    
    @ResponseBody
    @RequestMapping("/person/info")
    public Result info() {
        UserInfoVo userInfoVo = userService.getOtherUserInfo(UserInfo.getUserId());
        return Result.success().data("user", userInfoVo);
    }
    
    
    @RequestMapping("/detail")
    public String detail(@RequestParam String id, Model model) {
        //加入近期浏览
        historyUserWorkService.updateHistory(Long.parseLong(id));
        //引入相关信息
        model.addAttribute("id", id);
        return "detail";
    }
    @ResponseBody
    @PostMapping("/register")
    public Result register(@RequestParam String username,
                           @RequestParam String password,@RequestParam String intro) {
        UserRegisterVo user = new UserRegisterVo();
        user.setUsername(username);
        user.setPassword(password);
        user.setIntroduction(intro);
        boolean res = userService.register(user);
        return res ? Result.success().message("注册成功") : Result.error();
    }
 
    @GetMapping("/logout")
    public Result logout() {
        boolean res = userService.logout();
        return res ? Result.success().message("登出成功") : Result.error();
    }
    @ResponseBody
    @RequestMapping("/information")
    public Result changeInformation(@RequestBody UserRegisterVo user) {
        System.out.println("当前的值为"+user.getUsername()+"--"+user.getPassword()+"--"+user.getIntroduction());
        boolean res = userService.updateInformation(user);
        return res ? Result.success().message("个人信息更新成功") : Result.error();
    }
 
    @PutMapping("/privacy")
    public Result changePrivacy(@RequestBody UserPrivacyVo user) {
        boolean res = userService.updatePrivacy(user);
        return res ? Result.success().message("个人信息公开状态更新成功") : Result.error();
    }

 
    @GetMapping("/visit/privacy/{userId}")
    public Result searchPrivacySetting(@PathVariable Long userId) {
        UserPrivacyVo privacy = userService.getOtherPrivacySetting(userId);
        return Result.success().data("privacy", privacy);
    }
 
    @GetMapping("/visit/info/{userId}")
    public Result getInfo(@PathVariable Long userId) {
        UserInfoVo userInfoVo = userService.getOtherUserInfo(userId);
        return Result.success().data("user", userInfoVo);
    }
 
    @ResponseBody
    @GetMapping("/visit/get_id/{username}")
    public Result getUserIdByUsername(@PathVariable String username) {
        User userId = userService.getUserIdByUsername(username);
        return Result.success().data("user", userId);
    }
    @ResponseBody
    @PostMapping("/ban/{userId}")
    public Result reverseUserBanStatus(@PathVariable Long userId) {
        Boolean newBannedStatus = userService.reverseUserBanStatus(userId);
        return Result.success().data("isBanned", newBannedStatus);
    }
   
}
