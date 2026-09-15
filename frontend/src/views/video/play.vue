<template>
  <div class="app-container">
   <video controls autoplay :src="this.form.videoUrl"/>
  </div>
</template>
<script>
import videoApi from '@/api/video'
export default {
  data () {
    return {
      form: {
        videoId: null,
        userName: '',
        videoName: '',
        videoCategory: null,
        videoUrl: '',
        videoTagList: []
      },
      formLoading: false
    }
  },
  created () {
    let id = this.$route.query.id
    let _this = this
    if (id && parseInt(id) !== 0) {
      _this.formLoading = true
      videoApi.selectVideo(id).then(re => {
        _this.form = re.response
        _this.formLoading = false
      })
    }
  }
}
</script>
