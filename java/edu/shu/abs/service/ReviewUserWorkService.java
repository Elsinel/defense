package edu.shu.abs.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import edu.shu.abs.common.Page;
import edu.shu.abs.common.base.BaseService;
import edu.shu.abs.entity.ReviewUserWork;
import edu.shu.abs.vo.review.ReviewNewPostVo;
import edu.shu.abs.vo.review.ReviewQueryConditionVo;
import edu.shu.abs.vo.review.ReviewUserWorkWithUserVo;
import edu.shu.abs.vo.review.ReviewUserWorkWithWorkVo;

import javax.servlet.http.HttpSession;
import java.util.List;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
public interface ReviewUserWorkService extends BaseService<ReviewUserWork> {

    Page<ReviewUserWorkWithUserVo> getPage(int currentPage, int pageSize, long workId, ReviewQueryConditionVo condition);

    IPage<ReviewUserWorkWithWorkVo> getMyPageReview(int currentPage, int pageSize);
    
    IPage<ReviewUserWorkWithWorkVo> getAllPageReview(int currentPage, int pageSize);
    
    List<ReviewUserWorkWithWorkVo> getMyAllCollectionWithCheckingWork();

    ReviewUserWork getMyReview(long workId);

    boolean updateReview(ReviewNewPostVo review);

    boolean existRating(long workId, long userId);

    IPage<ReviewUserWorkWithWorkVo> getOtherPageReview(int currentPage, int pageSize, long userId);
}
