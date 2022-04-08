from model.swin.swin import Swin, EBM_Prior

def get_model(option):
    model_name = option['model_name']
    model = Swin(option['trainsize'], option['latent_dim']).cuda()
    ebm_model = EBM_Prior(option['ebm_out_dim'], option['ebm_middle_dim'], option['latent_dim']).cuda()
    print("Model based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(x.numel()/1e6 for x in model.parameters())))
    print("EBM Model based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(
        x.numel() / 1e6 for x in ebm_model.parameters())))
    return model, ebm_model
