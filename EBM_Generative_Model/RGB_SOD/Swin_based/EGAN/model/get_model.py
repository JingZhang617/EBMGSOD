from model.swin.swin import Swin, FCDiscriminator, EBM_Prior

def get_model(option):
    model_name = option['model_name']
    model = Swin(option['trainsize'],option['latent_dim']).cuda()
    discriminator = FCDiscriminator().cuda()
    ebm_model = EBM_Prior(option['ebm_out_dim'], option['ebm_middle_dim'], option['latent_dim']).cuda()
    print("Model based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(x.numel()/1e6 for x in model.parameters())))
    print("Discriminator based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(
        x.numel() / 1e6 for x in discriminator.parameters())))
    print("EBM based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(
        x.numel() / 1e6 for x in ebm_model.parameters())))


    return model, discriminator, ebm_model
