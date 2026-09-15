package com.alvis.media.viewmodel.video;

import com.alvis.media.domain.UserTag;
import com.alvis.media.domain.UserVideoOperation;
import com.alvis.media.domain.VideoPlay;
import lombok.Data;

import java.util.List;
@Data
public class VideoDetailVM {
    private Integer videoId;
    private String videoName;
    private List <VideoPlay> videoPlayList;



    private List <UserVideoOperation> videoOperationList;

    private List <UserTag> videoTagList;
}
