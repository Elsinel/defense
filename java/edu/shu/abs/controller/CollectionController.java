package edu.shu.abs.controller;

import edu.shu.abs.common.Result;
import edu.shu.abs.entity.Collection;
import edu.shu.abs.service.CollectionService;
import edu.shu.abs.vo.collection.CollectionNewPostVo;
import edu.shu.abs.vo.collection.CollectionUpdateVo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
@RestController
@RequestMapping("/collection")
public class CollectionController {
    @Autowired
    private CollectionService collectionService;

    @GetMapping("/my")
    public Result getMyAllCollection() {
        List<Collection> allCollection = collectionService.getMyAllCollection();
        return Result.success().data("all", allCollection);
    }

    @PostMapping("/my")
    public Result postNewCollection(@RequestBody CollectionNewPostVo collectionNewPostVo) {
        boolean res = collectionService.saveNewCollection(collectionNewPostVo);
        return res ? Result.success().message("收藏夹创建成功") : Result.error();
    }

    @PutMapping("/my")
    public Result updateCollection(@RequestBody CollectionUpdateVo collectionUpdateVo) {
        boolean res = collectionService.updateCollection(collectionUpdateVo);
        return res ? Result.success().message("收藏夹更新成功") : Result.error();
    }

    @DeleteMapping("/my/{collectionId}")
    public Result dropCollection(@PathVariable Long collectionId) {
        boolean res = collectionService.dropCollection(collectionId);
        return res ? Result.success().message("收藏夹删除成功") : Result.error();
    }

    @GetMapping("/visit/{userId}")
    public Result getMyAllCollection(@PathVariable long userId) {
        List<Collection> allCollection = collectionService.getOtherAllCollection(userId);
        return Result.success().data("all", allCollection);
    }
}
